package ml.util;

import ml.core.Tensor;
import ml.exception.ModelIOException;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import org.bytedeco.javacpp.DoublePointer;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;

public class ImageReader {

    public enum ColorMode { GRAY, RGB, BGR, RGBA }
    public enum Interpolation { NEAREST, LINEAR, CUBIC, LANCZOS }
    public enum FlipDirection { HORIZONTAL, VERTICAL, BOTH }

    public record ImageData(Mat mat, int width, int height, int channels) {}

    // ==================== READ ====================

    public static ImageData read(String filepath) throws ModelIOException {
        if (!Files.exists(Path.of(filepath))) throw new ModelIOException("File not found: " + filepath);
        Mat mat = imread(filepath, IMREAD_COLOR);
        if (mat == null || mat.empty()) throw new ModelIOException("Failed to read image: " + filepath);
        return new ImageData(mat, mat.cols(), mat.rows(), mat.channels());
    }

    public static ImageData readGrayscale(String filepath) throws ModelIOException {
        if (!Files.exists(Path.of(filepath))) throw new ModelIOException("File not found: " + filepath);
        Mat mat = imread(filepath, IMREAD_GRAYSCALE);
        if (mat == null || mat.empty()) throw new ModelIOException("Failed to read image: " + filepath);
        return new ImageData(mat, mat.cols(), mat.rows(), 1);
    }

    public static ImageData readWithAlpha(String filepath) throws ModelIOException {
        if (!Files.exists(Path.of(filepath))) throw new ModelIOException("File not found: " + filepath);
        Mat mat = imread(filepath, IMREAD_UNCHANGED);
        if (mat == null || mat.empty()) throw new ModelIOException("Failed to read image: " + filepath);
        return new ImageData(mat, mat.cols(), mat.rows(), mat.channels());
    }

    public static List<ImageData> readBatch(String... filepaths) throws ModelIOException {
        List<ImageData> images = new ArrayList<>();
        for (String fp : filepaths) images.add(read(fp));
        return images;
    }

    public static List<ImageData> readBatch(List<String> filepaths) throws ModelIOException {
        return readBatch(filepaths.toArray(new String[0]));
    }

    // ==================== WRITE ====================

    public static void write(ImageData image, String filepath) throws ModelIOException {
        String ext = getExtension(filepath);
        int[] params = getWriteParams(ext);
        boolean success = imwrite(filepath, image.mat(), params);
        if (!success) throw new ModelIOException("Failed to write image: " + filepath);
    }

    public static void write(Mat mat, String filepath) throws ModelIOException {
        write(new ImageData(mat, mat.cols(), mat.rows(), mat.channels()), filepath);
    }

    public static void writePng(ImageData image, String filepath) throws ModelIOException {
        boolean ok = imwrite(filepath, image.mat(), new int[]{IMWRITE_PNG_COMPRESSION, 6});
        if (!ok) throw new ModelIOException("Failed to write PNG: " + filepath);
    }

    public static void writeJpeg(ImageData image, String filepath, int quality) throws ModelIOException {
        boolean ok = imwrite(filepath, image.mat(), new int[]{IMWRITE_JPEG_QUALITY, quality});
        if (!ok) throw new ModelIOException("Failed to write JPEG: " + filepath);
    }

    // ==================== RESIZE ====================

    public static ImageData resize(ImageData image, int width, int height) {
        Mat resized = new Mat();
        org.bytedeco.opencv.global.opencv_imgproc.resize(image.mat(), resized, new Size(width, height), 0, 0, INTER_LINEAR);
        return new ImageData(resized, width, height, image.channels());
    }

    public static ImageData resize(ImageData image, int width, int height, Interpolation interp) {
        Mat resized = new Mat();
        int flag = switch (interp) {
            case NEAREST -> INTER_NEAREST;
            case LINEAR -> INTER_LINEAR;
            case CUBIC -> INTER_CUBIC;
            case LANCZOS -> INTER_LANCZOS4;
        };
        org.bytedeco.opencv.global.opencv_imgproc.resize(image.mat(), resized, new Size(width, height), 0, 0, flag);
        return new ImageData(resized, width, height, image.channels());
    }

    public static ImageData resizeToTensorSize(ImageData image, int targetSize) {
        return resize(image, targetSize, targetSize);
    }

    public static ImageData resizeShortestSide(ImageData image, int targetSize) {
        int w = image.width();
        int h = image.height();
        double scale = (double) targetSize / Math.min(w, h);
        int newW = (int) (w * scale);
        int newH = (int) (h * scale);
        return resize(image, newW, newH);
    }

    // ==================== CROP ====================

    public static ImageData crop(ImageData image, int x, int y, int width, int height) {
        Rect roi = new Rect(x, y, width, height);
        Mat cropped = new Mat(image.mat(), roi).clone();
        return new ImageData(cropped, width, height, image.channels());
    }

    public static ImageData centerCrop(ImageData image, int size) {
        int w = image.width();
        int h = image.height();
        int x = (w - size) / 2;
        int y = (h - size) / 2;
        return crop(image, Math.max(0, x), Math.max(0, y), size, size);
    }

    public static ImageData randomCrop(ImageData image, int size, Random rng) {
        int w = image.width();
        int h = image.height();
        int x = rng.nextInt(Math.max(1, w - size + 1));
        int y = rng.nextInt(Math.max(1, h - size + 1));
        return crop(image, x, y, size, size);
    }

    // ==================== FLIP ====================

    public static ImageData flip(ImageData image, FlipDirection direction) {
        Mat flipped = new Mat();
        int code = switch (direction) {
            case HORIZONTAL -> 1;
            case VERTICAL -> 0;
            case BOTH -> -1;
        };
        org.bytedeco.opencv.global.opencv_core.flip(image.mat(), flipped, code);
        return new ImageData(flipped, image.width(), image.height(), image.channels());
    }

    public static ImageData flipHorizontal(ImageData image) { return flip(image, FlipDirection.HORIZONTAL); }
    public static ImageData flipVertical(ImageData image) { return flip(image, FlipDirection.VERTICAL); }

    // ==================== ROTATE ====================

    public static ImageData rotate(ImageData image, double angleDegrees) {
        Point2f center = new Point2f(image.width() / 2f, image.height() / 2f);
        Mat rotMat = getRotationMatrix2D(center, angleDegrees, 1.0);
        Mat rotated = new Mat();
        org.bytedeco.opencv.global.opencv_imgproc.warpAffine(image.mat(), rotated, rotMat, new Size(image.width(), image.height()));
        return new ImageData(rotated, image.width(), image.height(), image.channels());
    }

    public static ImageData rotate(ImageData image, double angleDegrees, Interpolation interp) {
        Point2f center = new Point2f(image.width() / 2f, image.height() / 2f);
        Mat rotMat = getRotationMatrix2D(center, angleDegrees, 1.0);
        Mat rotated = new Mat();
        int flag = switch (interp) {
            case NEAREST -> INTER_NEAREST;
            case LINEAR -> INTER_LINEAR;
            case CUBIC -> INTER_CUBIC;
            case LANCZOS -> INTER_LANCZOS4;
        };
        org.bytedeco.opencv.global.opencv_imgproc.warpAffine(image.mat(), rotated, rotMat,
            new Size(image.width(), image.height()), flag, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));
        return new ImageData(rotated, image.width(), image.height(), image.channels());
    }

    public static ImageData rotate90(ImageData image) {
        Mat rotated = new Mat();
        transpose(image.mat(), rotated);
        org.bytedeco.opencv.global.opencv_core.flip(rotated, rotated, 1);
        return new ImageData(rotated, image.height(), image.width(), image.channels());
    }

    public static ImageData rotate180(ImageData image) {
        Mat rotated = new Mat();
        org.bytedeco.opencv.global.opencv_core.flip(image.mat(), rotated, -1);
        return new ImageData(rotated, image.width(), image.height(), image.channels());
    }

    public static ImageData rotate270(ImageData image) {
        Mat rotated = new Mat();
        transpose(image.mat(), rotated);
        org.bytedeco.opencv.global.opencv_core.flip(rotated, rotated, 0);
        return new ImageData(rotated, image.height(), image.width(), image.channels());
    }

    // ==================== ZOOM ====================

    public static ImageData zoom(ImageData image, double factor) {
        int newW = (int) (image.width() * factor);
        int newH = (int) (image.height() * factor);
        ImageData resized = resize(image, newW, newH);

        int x = (newW - image.width()) / 2;
        int y = (newH - image.height()) / 2;
        return crop(resized, Math.max(0, x), Math.max(0, y), image.width(), image.height());
    }

    // ==================== COLOR CONVERSION ====================

    public static ImageData toGrayscale(ImageData image) {
        if (image.channels() == 1) return image;
        Mat gray = new Mat();
        cvtColor(image.mat(), gray, COLOR_BGR2GRAY);
        return new ImageData(gray, image.width(), image.height(), 1);
    }

    public static ImageData toRGB(ImageData image) {
        if (image.channels() == 3) return image;
        Mat rgb = new Mat();
        if (image.channels() == 1) {
            cvtColor(image.mat(), rgb, COLOR_GRAY2RGB);
        } else if (image.channels() == 4) {
            cvtColor(image.mat(), rgb, COLOR_BGRA2RGB);
        }
        return new ImageData(rgb, image.width(), image.height(), 3);
    }

    public static ImageData toBGR(ImageData image) {
        if (image.channels() == 3) return image;
        Mat bgr = new Mat();
        if (image.channels() == 1) {
            cvtColor(image.mat(), bgr, COLOR_GRAY2BGR);
        } else if (image.channels() == 4) {
            cvtColor(image.mat(), bgr, COLOR_BGRA2BGR);
        }
        return new ImageData(bgr, image.width(), image.height(), 3);
    }

    public static ImageData toRGBA(ImageData image) {
        if (image.channels() == 4) return image;
        Mat rgba = new Mat();
        if (image.channels() == 1) {
            cvtColor(image.mat(), rgba, COLOR_GRAY2RGBA);
        } else if (image.channels() == 3) {
            cvtColor(image.mat(), rgba, COLOR_BGR2RGBA);
        }
        return new ImageData(rgba, image.width(), image.height(), 4);
    }

    // ==================== NORMALIZATION ====================

    public static ImageData normalize(ImageData image, double min, double max) {
        Mat normalized = new Mat();
        image.mat().convertTo(normalized, CV_64F, 1.0 / 255.0, 0.0);
        return new ImageData(normalized, image.width(), image.height(), image.channels());
    }

    public static ImageData normalizeMeanStd(ImageData image, double[] mean, double[] std) {
        int w = image.width();
        int h = image.height();
        int c = image.channels();
        int total = w * h * c;

        byte[] buf = new byte[total];
        image.mat().data().get(buf);

        double[] out = new double[total];
        for (int i = 0; i < total; i++) {
            int ch = (c == 1) ? 0 : i % c;
            out[i] = ((buf[i] & 0xFF) / 255.0 - mean[ch]) / std[ch];
        }

        int type = switch (c) {
            case 3 -> CV_64FC3;
            case 4 -> CV_64FC4;
            default -> CV_64FC1;
        };
        Mat tmp = new Mat(new Size(w, h), type, new DoublePointer(out));
        return new ImageData(tmp.clone(), w, h, c);
    }

    // ==================== AUGMENTATION ====================

    public static ImageData randomFlip(ImageData image, double probability, Random rng) {
        if (rng.nextDouble() > probability) return image;
        return flipHorizontal(image);
    }

    public static ImageData randomRotation(ImageData image, double maxAngle, double probability, Random rng) {
        if (rng.nextDouble() > probability) return image;
        double angle = (rng.nextDouble() * 2 - 1) * maxAngle;
        return rotate(image, angle);
    }

    public static ImageData randomCrop(ImageData image, int size, double probability, Random rng) {
        if (rng.nextDouble() > probability) return resize(image, size, size);
        return randomCrop(image, size, rng);
    }

    public static ImageData randomZoom(ImageData image, double zoomRange, double probability, Random rng) {
        if (rng.nextDouble() > probability) return image;
        double factor = 1.0 + (rng.nextDouble() * 2 - 1) * zoomRange;
        return zoom(image, factor);
    }

    // ==================== TENSOR CONVERSION ====================

    public static Tensor toTensor(ImageData image) {
        return toTensor(image, ColorMode.RGB, true);
    }

    public static Tensor toTensor(ImageData image, ColorMode mode, boolean normalize) {
        ImageData converted = switch (mode) {
            case GRAY -> toGrayscale(image);
            case RGB -> toRGB(image);
            case BGR -> toBGR(image);
            case RGBA -> toRGBA(image);
        };

        int channels = converted.channels();
        int h = converted.height();
        int w = converted.width();
        int pixels = h * w;

        double[] data = new double[channels * pixels];
        byte[] buffer = new byte[channels * pixels];
        converted.mat().data().get(buffer);

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int bufIdx = (y * w + x) * channels + c;
                    int dataIdx = c * pixels + y * w + x;
                    data[dataIdx] = normalize ? (buffer[bufIdx] & 0xFF) / 255.0 : (buffer[bufIdx] & 0xFF);
                }
            }
        }

        return Tensor.fromFlat(data, 1, channels * pixels);
    }

    public static Tensor batchToTensor(List<ImageData> images, ColorMode mode, boolean normalize) {
        if (images.isEmpty()) throw new IllegalArgumentException("Empty image list");

        int channels = switch (mode) {
            case GRAY -> 1;
            case RGB, BGR -> 3;
            case RGBA -> 4;
        };

        int h = images.get(0).height();
        int w = images.get(0).width();
        int batchSize = images.size();
        int pixels = h * w;

        float[] allData = new float[batchSize * channels * pixels];

        for (int b = 0; b < batchSize; b++) {
            Tensor t = toTensor(images.get(b), mode, normalize);
            System.arraycopy(t.data(), 0, allData, b * channels * pixels, channels * pixels);
        }

        return Tensor.fromFlat(allData, batchSize, channels * pixels);
    }

    public static Tensor batchToTensor(String... filepaths) throws ModelIOException {
        List<ImageData> images = readBatch(filepaths);
        return batchToTensor(images, ColorMode.RGB, true);
    }

    public static Tensor batchToTensorGrayscale(String... filepaths) throws ModelIOException {
        List<ImageData> images = new ArrayList<>();
        for (String fp : filepaths) images.add(readGrayscale(fp));
        return batchToTensor(images, ColorMode.GRAY, true);
    }

    // ==================== TENSOR TO IMAGE ====================

    public static ImageData tensorToImage(Tensor tensor, int width, int height) {
        return tensorToImage(tensor, width, height, ColorMode.RGB, true);
    }

    public static ImageData tensorToImage(Tensor tensor, int width, int height, ColorMode mode, boolean normalized) {
        int channels = switch (mode) {
            case GRAY -> 1;
            case RGB, BGR -> 3;
            case RGBA -> 4;
        };

        int pixels = width * height;
        byte[] buffer = new byte[channels * pixels];
        float[] data = tensor.data();

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int dataIdx = c * pixels + y * width + x;
                    int bufIdx = (y * width + x) * channels + c;
                    double val = data[dataIdx];
                    if (normalized) val *= 255.0;
                    buffer[bufIdx] = (byte) Math.max(0, Math.min(255, (int) val));
                }
            }
        }

        int type = switch (mode) {
            case GRAY -> CV_8UC1;
            case RGB -> CV_8UC3;
            case BGR -> CV_8UC3;
            case RGBA -> CV_8UC4;
        };

        Mat mat = new Mat(height, width, type);
        mat.data().put(buffer);
        return new ImageData(mat, width, height, channels);
    }

    public static List<ImageData> batchTensorToImages(Tensor batch, int width, int height, ColorMode mode, boolean normalized) {
        List<ImageData> images = new ArrayList<>();
        int channels = switch (mode) {
            case GRAY -> 1;
            case RGB, BGR -> 3;
            case RGBA -> 4;
        };
        int pixelsPerImage = channels * width * height;

        for (int b = 0; b < batch.rows(); b++) {
            float[] imgData = new float[pixelsPerImage];
            System.arraycopy(batch.data(), b * pixelsPerImage, imgData, 0, pixelsPerImage);
            Tensor single = Tensor.fromFlat(imgData, 1, pixelsPerImage);
            images.add(tensorToImage(single, width, height, mode, normalized));
        }
        return images;
    }

    // ==================== DIRECTORY OPERATIONS ====================

    public static List<ImageData> readDirectory(String dirPath, String extension) throws ModelIOException {
        File dir = new File(dirPath);
        if (!dir.isDirectory()) throw new ModelIOException("Not a directory: " + dirPath);

        List<ImageData> images = new ArrayList<>();
        File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith("." + extension.toLowerCase()));
        if (files != null) {
            Arrays.sort(files);
            for (File f : files) {
                images.add(read(f.getAbsolutePath()));
            }
        }
        return images;
    }

    public static Tensor readDirectoryAsTensor(String dirPath, int targetSize, ColorMode mode) throws ModelIOException {
        File dir = new File(dirPath);
        if (!dir.isDirectory()) throw new ModelIOException("Not a directory: " + dirPath);

        String[] extensions = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"};
        List<File> imageFiles = new ArrayList<>();
        for (String ext : extensions) {
            File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith("." + ext));
            if (files != null) imageFiles.addAll(Arrays.asList(files));
        }
        imageFiles.sort(Comparator.comparing(File::getName));

        List<ImageData> images = new ArrayList<>();
        for (File f : imageFiles) {
            ImageData img = read(f.getAbsolutePath());
            ImageData resized = resizeToTensorSize(img, targetSize);
            images.add(resized);
        }

        return batchToTensor(images, mode, true);
    }

    public static int countImages(String dirPath) {
        File dir = new File(dirPath);
        if (!dir.isDirectory()) return 0;
        String[] extensions = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"};
        int count = 0;
        for (String ext : extensions) {
            File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith("." + ext));
            if (files != null) count += files.length;
        }
        return count;
    }

    // ==================== INFO ====================

    public static void printImageInfo(ImageData image) {
        System.out.printf("Image: %dx%d, %d channels, type=%s%n",
            image.width(), image.height(), image.channels(), matTypeString(image.mat()));
    }

    private static String matTypeString(Mat mat) {
        return switch (mat.type()) {
            case 0 -> "CV_8UC1";
            case 1 -> "CV_8UC2";
            case 2 -> "CV_8UC3";
            case 3 -> "CV_8UC4";
            case 5 -> "CV_16UC1";
            case 6 -> "CV_16UC3";
            case 11 -> "CV_32FC1";
            case 12 -> "CV_32FC2";
            case 13 -> "CV_32FC3";
            case 14 -> "CV_32FC4";
            case 21 -> "CV_64FC1";
            case 22 -> "CV_64FC2";
            case 23 -> "CV_64FC3";
            case 24 -> "CV_64FC4";
            default -> "unknown(" + mat.type() + ")";
        };
    }

    // ==================== HELPERS ====================

    private static String getExtension(String filepath) {
        int dot = filepath.lastIndexOf('.');
        return dot >= 0 ? filepath.substring(dot + 1).toLowerCase() : "";
    }

    private static int[] getWriteParams(String ext) {
        return switch (ext) {
            case "jpg", "jpeg" -> new int[]{IMWRITE_JPEG_QUALITY, 95};
            case "png" -> new int[]{IMWRITE_PNG_COMPRESSION, 6};
            case "tiff", "tif" -> new int[]{IMWRITE_TIFF_COMPRESSION, 5};
            default -> new int[]{};
        };
    }

    // ==================== GAUSSIAN BLUR / SHARPEN ====================

    public static ImageData gaussianBlur(ImageData image, int kernelSize, double sigma) {
        Mat blurred = new Mat();
        GaussianBlur(image.mat(), blurred, new Size(kernelSize, kernelSize), sigma);
        return new ImageData(blurred, image.width(), image.height(), image.channels());
    }

    public static ImageData sharpen(ImageData image) {
        Mat sharpened = new Mat();
        double[] kernelData = {
            0, -1, 0,
           -1,  5, -1,
            0, -1, 0
        };
        Mat kernel = new Mat(new Size(3, 3), CV_64F, new DoublePointer(kernelData));
        filter2D(image.mat(), sharpened, -1, kernel);
        return new ImageData(sharpened, image.width(), image.height(), image.channels());
    }

    public static ImageData brightness(ImageData image, double factor) {
        Mat result = new Mat();
        image.mat().convertTo(result, -1, factor, 0);
        return new ImageData(result, image.width(), image.height(), image.channels());
    }

    public static ImageData contrast(ImageData image, double factor) {
        Mat result = new Mat();
        image.mat().convertTo(result, -1, factor, 127.5 * (1 - factor));
        return new ImageData(result, image.width(), image.height(), image.channels());
    }

    public static ImageData equalizeHistogram(ImageData image) {
        if (image.channels() != 1) {
            ImageData gray = toGrayscale(image);
            Mat equalized = new Mat();
            equalizeHist(gray.mat(), equalized);
            return new ImageData(equalized, gray.width(), gray.height(), 1);
        }
        Mat equalized = new Mat();
        equalizeHist(image.mat(), equalized);
        return new ImageData(equalized, image.width(), image.height(), 1);
    }

    public static ImageData threshold(ImageData image, double thresh, double maxVal) {
        Mat result = new Mat();
        org.bytedeco.opencv.global.opencv_imgproc.threshold(image.mat(), result, thresh, maxVal, THRESH_BINARY);
        return new ImageData(result, image.width(), image.height(), image.channels());
    }

    public static ImageData cannyEdge(ImageData image, double low, double high) {
        Mat gray = image.channels() == 1 ? image.mat() : toGrayscale(image).mat();
        Mat edges = new Mat();
        Canny(gray, edges, low, high);
        return new ImageData(edges, image.width(), image.height(), 1);
    }

    public static ImageData dilate(ImageData image, int kernelSize) {
        Mat dilated = new Mat();
        Mat kernel = getStructuringElement(MORPH_RECT, new Size(kernelSize, kernelSize));
        org.bytedeco.opencv.global.opencv_imgproc.dilate(image.mat(), dilated, kernel);
        return new ImageData(dilated, image.width(), image.height(), image.channels());
    }

    public static ImageData erode(ImageData image, int kernelSize) {
        Mat eroded = new Mat();
        Mat kernel = getStructuringElement(MORPH_RECT, new Size(kernelSize, kernelSize));
        org.bytedeco.opencv.global.opencv_imgproc.erode(image.mat(), eroded, kernel);
        return new ImageData(eroded, image.width(), image.height(), image.channels());
    }

    public static ImageData morphOpen(ImageData image, int kernelSize) {
        Mat opened = new Mat();
        Mat kernel = getStructuringElement(MORPH_RECT, new Size(kernelSize, kernelSize));
        morphologyEx(image.mat(), opened, MORPH_OPEN, kernel);
        return new ImageData(opened, image.width(), image.height(), image.channels());
    }

    public static ImageData morphClose(ImageData image, int kernelSize) {
        Mat closed = new Mat();
        Mat kernel = getStructuringElement(MORPH_RECT, new Size(kernelSize, kernelSize));
        morphologyEx(image.mat(), closed, MORPH_CLOSE, kernel);
        return new ImageData(closed, image.width(), image.height(), image.channels());
    }

    public static ImageData medianBlur(ImageData image, int kernelSize) {
        Mat blurred = new Mat();
        org.bytedeco.opencv.global.opencv_imgproc.medianBlur(image.mat(), blurred, kernelSize);
        return new ImageData(blurred, image.width(), image.height(), image.channels());
    }

    public static ImageData applyBilateralFilter(ImageData image, int d, double sigmaColor, double sigmaSpace) {
        Mat filtered = new Mat();
        org.bytedeco.opencv.global.opencv_imgproc.bilateralFilter(image.mat(), filtered, d, sigmaColor, sigmaSpace);
        return new ImageData(filtered, image.width(), image.height(), image.channels());
    }
}
