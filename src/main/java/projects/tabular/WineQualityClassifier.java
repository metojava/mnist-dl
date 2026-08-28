package projects.tabular;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import projects.common.Proj;

import java.util.Random;

/**
 * P13 - Wine quality classification (low/medium/high) from physicochemical features.
 * Generates a realistic synthetic wine-chemistry dataset; if the real UCI file exists at
 * data/wine/winequality-red.csv it is loaded instead (semicolon-separated, header row).
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.tabular.WineQualityClassifier" "-Dexec.args=40 32"
 */
public class WineQualityClassifier {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 40), batch = Proj.batch(args, 32);
        long t0 = System.currentTimeMillis();

        Tensor x;
        int[] y;
        java.nio.file.Path csv = java.nio.file.Path.of("data/wine/winequality-red.csv");
        if (java.nio.file.Files.exists(csv)) {
            var res = ml.util.DataReader.readCsv(csv.toString(), true, ";");
            x = res.features(); y = binQuality(res.labels());
            System.out.println("[P13 Wine] loaded real UCI dataset");
        } else {
            int n = 1600;
            x = new Tensor(n, 7);
            y = new int[n];
            Random rng = new Random(42);
            for (int i = 0; i < n; i++) {
                double fixedAcid = 6 + 2 * rng.nextGaussian();
                double volatileAcid = 0.35 + 0.12 * rng.nextGaussian();
                double citric = Math.max(0, 0.28 + 0.12 * rng.nextGaussian());
                double sulphates = 0.6 + 0.13 * rng.nextGaussian();
                double alcohol = 9.5 + 1.1 * rng.nextGaussian();
                double ph = 3.2 + 0.12 * rng.nextGaussian();
                double sugar = 2.4 + 1.0 * rng.nextGaussian();

                x.set(i, 0, fixedAcid); x.set(i, 1, -volatileAcid); x.set(i, 2, citric);
                x.set(i, 3, sulphates); x.set(i, 4, alcohol); x.set(i, 5, ph); x.set(i, 6, -sugar);

                double score = 0.30 * alcohol + 1.6 * sulphates + 1.2 * citric
                    - 2.5 * volatileAcid - 0.15 * (sugar - 2.4) + 0.25 * rng.nextGaussian() + 2.2;
                y[i] = score < 4.6 ? 0 : (score < 5.8 ? 1 : 2);
            }
            System.out.println("[P13 Wine] using synthetic chemistry dataset (drop the UCI csv into data/wine to use real data)");
        }

        Proj.standardize(x);
        Proj.shuffle(x, y, new java.util.Random(42));
        String[] names = {"low", "medium", "high"};

        int split = (int) (x.rows() * 0.75);
        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(x.cols(), 48));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(48, 24));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(24, 3));
        net.add(new ml.activation.Softmax());

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.005))
            .fit(x.sliceRows(0, split), java.util.Arrays.copyOf(y, split), epochs, batch);

        System.out.printf("Train acc: %.1f%% | Test acc: %.1f%%%n",
            100 * acc(net, x.sliceRows(0, split), java.util.Arrays.copyOf(y, split)),
            100 * acc(net, x.sliceRows(split, x.rows()), java.util.Arrays.copyOfRange(y, split, x.rows())));
        Proj.footer("P13 Wine", t0);
    }

    private static int[] binQuality(int[] q) {
        return java.util.Arrays.stream(q).map(v -> v <= 4 ? 0 : (v <= 6 ? 1 : 2)).toArray();
    }

    private static double acc(NeuralNetwork net, Tensor x, int[] y) {
        Tensor out = net.forward(x, false);
        int c = 0;
        for (int i = 0; i < x.rows(); i++) if (out.argmaxRow(i) == y[i]) c++;
        return (double) c / x.rows();
    }
}
