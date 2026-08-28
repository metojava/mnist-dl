package ml.util;

import ml.core.Tensor;
import ml.exception.ModelIOException;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MNISTLoader {
    private static final int IMAGE_MAGIC = 2051;
    private static final int LABEL_MAGIC = 2049;

    public static ml.training.Dataset load(String imageFile, String labelFile) throws ModelIOException {
        long start = System.currentTimeMillis();
        Tensor images = loadImages(imageFile);
        int[] labels = loadLabels(labelFile);
        long elapsed = System.currentTimeMillis() - start;
        System.out.printf("Loaded %d images in %dms%n", images.rows(), elapsed);
        return new ml.training.Dataset(images, labels);
    }

    private static Tensor loadImages(String filepath) throws ModelIOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filepath), 1024 * 1024))) {
            int magic = readInt(dis);
            if (magic != IMAGE_MAGIC)
                throw new ModelIOException("Invalid image magic number: " + magic + " (expected " + IMAGE_MAGIC + ")");
            int count = readInt(dis);
            int rows = readInt(dis);
            int cols = readInt(dis);
            int pixels = rows * cols;
            Tensor images = new Tensor(count, pixels);
            byte[] buffer = new byte[count * pixels];
            dis.readFully(buffer);
            for (int i = 0; i < buffer.length; i++)
                images.data()[i] = (buffer[i] & 0xFF) / 255.0;
            return images;
        } catch (IOException e) {
            throw new ModelIOException("Failed to load images from " + filepath + ": " + e.getMessage(), e);
        }
    }

    private static int[] loadLabels(String filepath) throws ModelIOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filepath)))) {
            int magic = readInt(dis);
            if (magic != LABEL_MAGIC)
                throw new ModelIOException("Invalid label magic number: " + magic + " (expected " + LABEL_MAGIC + ")");
            int count = readInt(dis);
            int[] labels = new int[count];
            for (int i = 0; i < count; i++) labels[i] = dis.readUnsignedByte();
            return labels;
        } catch (IOException e) {
            throw new ModelIOException("Failed to load labels from " + filepath + ": " + e.getMessage(), e);
        }
    }

    private static int readInt(DataInputStream dis) throws IOException {
        byte[] bytes = new byte[4];
        dis.readFully(bytes);
        return ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).getInt();
    }
}
