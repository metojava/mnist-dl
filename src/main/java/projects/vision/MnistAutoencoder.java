package projects.vision;

import ml.core.Tensor;
import projects.common.Proj;
import ml.core.SeedManager;
import projects.common.Proj;
import ml.network.NeuralNetwork;
import projects.common.Proj;
import ml.training.*;
import projects.common.Proj;
import ml.optimizer.Adam;
import projects.common.Proj;
import ml.loss.MSELoss;
import projects.common.Proj;
import ml.util.MNISTLoader;
import projects.common.Proj;

import javax.imageio.ImageIO;
import projects.common.Proj;
import java.awt.*;
import projects.common.Proj;
import java.awt.image.BufferedImage;
import projects.common.Proj;
import java.io.File;
import projects.common.Proj;
import java.util.Random;
import projects.common.Proj;

/**
 * P04 - Denoising autoencoder on MNIST.
 * Corrupts digits with salt-and-pepper noise and learns to reconstruct the clean image
 * (784 -> 256 -> 64 -> 256 -> 784). Saves a noisy/reconstructed grid PNG.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.vision.MnistAutoencoder" "-Dexec.args=10 256"
 */
public class MnistAutoencoder {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 10), batch = Proj.batch(args, 256);
        var out = Proj.out("autoencoder");
        long t0 = System.currentTimeMillis();

        Dataset train = ml.util.MNISTLoader.load(
            "data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte");

        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(784, 256));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(256, 64));   // bottleneck
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(64, 256));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(256, 784));
        net.add(new ml.activation.Sigmoid());
        System.out.printf("[P04 Autoencoder] params=%d%n", net.totalParameters());

        Random rng = new Random(7);
        Trainer trainer = new Trainer(net, new MSELoss(), new Adam(0.002));

        for (int ep = 1; ep <= epochs; ep++) {
            double lossSum = 0; int batches = 0;
            var iter = new ml.training.DataLoader(batch).iterate(train.images(), train.labels());
            while (iter.hasNext()) {
                var b = iter.next();
                Tensor noisy = addNoise(b.images(), rng);
                Tensor pred = net.forward(noisy, true);
                double loss = new MSELoss().forward(pred, b.labels());   // labels unused by MSE
                // target = CLEAN images: compute grad manually (MSE vs clean)
                Tensor grad = pred.subtract(b.images()).multiply(2.0 / (batchSize(b)));
                net.backward(grad);
                net.update(0.002);
                lossSum += mse(pred, b.images()); batches++;
                if (loss != loss) break;
            }
            System.out.printf("Epoch %d/%d - recon MSE: %.6f%n", ep, epochs, lossSum / Math.max(1, batches));
        }

        saveGrid(net, train, out.resolve("denoise_grid.png").toString(), rng);
        net.saveModel(out.resolve("model.bin").toString());
        System.out.println("Artifacts: " + out.toAbsolutePath());
        Proj.footer("P04 Autoencoder", t0);
    }

    private static int batchSize(ml.training.DataLoader.Batch b) { return b.images().rows(); }

    private static Tensor addNoise(Tensor x, Random rng) {
        Tensor n = x.copy();
        for (int i = 0; i < n.data().length; i++)
            if (rng.nextDouble() < 0.15) n.data()[i] = (float) (rng.nextBoolean() ? 1.0 : 0.0);
        return n;
    }

    private static double mse(Tensor a, Tensor b) {
        double s = 0;
        for (int i = 0; i < a.size(); i++) { double d = a.data()[i] - b.data()[i]; s += d * d; }
        return s / a.size();
    }

    private static void saveGrid(NeuralNetwork net, Dataset data, String path, Random rng) throws Exception {
        int n = 6;
        Tensor sample = data.images().sliceRows(0, n);
        Tensor noisy = addNoise(sample, rng);
        Tensor recon = net.forward(noisy, false);

        int cell = 112, pad = 8;
        BufferedImage img = new BufferedImage(n * cell + pad * (n + 1), 3 * cell + pad * 4,
            BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE); g.fillRect(0, 0, img.getWidth(), img.getHeight());
        for (int i = 0; i < n; i++) {
            draw(g, toImage(sample, i), pad + i * (cell + pad), pad, cell, "clean");
            draw(g, toImage(noisy, i), pad + i * (cell + pad), cell + pad * 2, cell, "noisy");
            draw(g, toImage(recon, i), pad + i * (cell + pad), 2 * cell + pad * 3, cell, "recon");
        }
        g.dispose();
        ImageIO.write(img, "png", new File(path));
        System.out.println("Saved grid: " + path);
    }

    private static BufferedImage toImage(Tensor t, int row) {
        BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++) {
                int v = (int) (Math.max(0, Math.min(1, t.get(row, y * 28 + x))) * 255);
                bi.getRaster().setSample(x, y, 0, v);
            }
        return bi;
    }

    private static void draw(Graphics2D g, BufferedImage src, int x, int y, int size, String tag) {
        g.drawImage(src, x, y, size, size, null);
        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font("Arial", Font.PLAIN, 11));
        g.drawString(tag, x + 2, y + size - 3);
    }
}

