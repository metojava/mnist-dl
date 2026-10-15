package projects.vision;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import ml.layer.Conv2D;
import ml.layer.MaxPool;
import ml.regularization.Dropout;
import ml.util.ImageReader;
import projects.common.Proj;

import java.io.File;
import java.util.*;

/**
 * P03 - CIFAR-100 image classification from the ImageFolder layout
 *       (data/CIFAR-100/{train,test}/<class_name>/*.png), using a small CNN.
 *
 * Loads images through JavaCV/OpenCV (ImageReader) and trains:
 *   Conv(3->16,3x3)-ReLU-MaxPool - Conv(16->32,3x3)-ReLU-MaxPool - Dense-Softmax
 *
 * Run  : mvn compile exec:java "-Dexec.mainClass=projects.vision.Cifar100Classifier"
 *        "-Dexec.args=8 64 120"      (epochs, batch, per-class limit; default 120 => ~12k imgs)
 */
public class Cifar100Classifier {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 8), batch = Proj.batch(args, 64);
        int perClass = Proj.limit(args, 120);
        long t0 = System.currentTimeMillis();

        System.out.printf("[P03 CIFAR-100] loading (per-class cap=%d)...%n", perClass);
        Dataset train = loadFolder("data/CIFAR-100/train", perClass, 42);
        Dataset test = loadFolder("data/CIFAR-100/test", Math.min(perClass, 50), 7);
        int classes = train.classes();
        System.out.printf("train=%d test=%d classes=%d%n", train.size(), test.size(), classes);

        NeuralNetwork net = new NeuralNetwork();
        net.add(new Conv2D(3, 16, 3));
        net.add(new ml.activation.ReLU());
        net.add(new MaxPool(2, 2, 16));                 // 30 -> 15
        net.add(new Conv2D(16, 32, 3));
        net.add(new ml.activation.ReLU());
        net.add(new MaxPool(2, 2, 32));                 // 13 -> 6
        net.add(new ml.layer.DenseLayer(32 * 6 * 6, 256));
        net.add(new ml.activation.ReLU());
        net.add(new Dropout(0.4));
        net.add(new ml.layer.DenseLayer(256, classes));
        net.add(new ml.activation.Softmax());
        System.out.printf("params=%d%n", net.totalParameters());

        Trainer trainer = new Trainer(net, new CrossEntropyLoss(), new Adam(0.002));
        trainer.fit(train.images(), train.labels(), epochs, batch);

        Tensor testOut = net.forward(test.images(), false);
        int correct = 0;
        for (int i = 0; i < testOut.rows(); i++) if (testOut.argmaxRow(i) == test.labels()[i]) correct++;
        System.out.printf("Test accuracy: %.2f%% (%d/%d)%n", 100.0 * correct / test.size(), correct, test.size());
        net.saveModel(Proj.out("cifar100").resolve("model.bin").toString());
        Proj.footer("P03 CIFAR-100", t0);
    }

    /** Reads an ImageFolder tree into a Dataset with class-interleaved rows. */
    private static Dataset loadFolder(String root, int perClass, long seed) {
        File[] classDirs = new File(root).listFiles(File::isDirectory);
        Objects.requireNonNull(classDirs, "missing " + root);
        Arrays.sort(classDirs);

        List<List<File>> byClass = new ArrayList<>();
        List<String> classNames = new ArrayList<>();
        for (File d : classDirs) {
            File[] imgs = d.listFiles((x, n) -> n.toLowerCase().matches(".*\\.(png|jpg|jpeg|bmp)$"));
            Arrays.sort(imgs != null ? imgs : new File[0]);
            List<File> list = new ArrayList<>();
            if (imgs != null) list.addAll(Arrays.asList(imgs).subList(0, Math.min(perClass, imgs.length)));
            if (!list.isEmpty()) { byClass.add(list); classNames.add(d.getName()); }
        }

        Random rng = new Random(seed);
        List<int[]> positions = new ArrayList<>(); // {classIdx, fileIdx}
        int total = 0;
        for (int c = 0; c < byClass.size(); c++) {
            for (int f = 0; f < byClass.get(c).size(); f++) { positions.add(new int[]{c, f}); total++; }
        }
        java.util.Collections.shuffle(positions, rng);   // de-cluster classes

        float[] data = new float[total * 3 * 32 * 32];
        int[] labels = new int[total];
        int row = 0;
        for (int[] pos : positions) {
            File img = byClass.get(pos[0]).get(pos[1]);
            ImageReader.ImageData im = ImageReader.read(img.getPath());
            ImageReader.ImageData r32 = ImageReader.resize(im, 32, 32);
            Tensor t = ImageReader.toTensor(r32, ImageReader.ColorMode.RGB, true);
            System.arraycopy(t.data(), 0, data, row * t.cols(), t.cols());
            labels[row] = pos[0];
            row++;
        }
        if (row % 2000 == 0 || row == total) System.out.printf("  loaded %d/%d%n", row, total);
        return new Dataset(Tensor.fromFlat(data, total, 3 * 32 * 32), labels);
    }
}

