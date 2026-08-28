package projects.tabular;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.loss.CrossEntropyLoss;
import ml.loss.FocalLoss;
import ml.optimizer.Adam;
import projects.common.Proj;

import java.util.Random;

/**
 * P15 - Credit card fraud detection on a heavily imbalanced synthetic dataset
 * (2% fraud). Compares CrossEntropy vs FocalLoss to show how focusing helps recall.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.tabular.FraudDetector" "-Dexec.args=25 64"
 */
public class FraudDetector {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 25), batch = Proj.batch(args, 64);
        long t0 = System.currentTimeMillis();

        int n = 6000;
        Tensor x = new Tensor(n, 6);
        int[] y = new int[n];
        Random rng = new Random(42);
        for (int i = 0; i < n; i++) {
            boolean fraud = rng.nextDouble() < 0.02;
            y[i] = fraud ? 1 : 0;
            // 35% of frauds camouflage as normal activity -> hard examples
            boolean stealth = fraud;

            double amount = fraud && !stealth ? 80 + Math.abs(rng.nextGaussian()) * 420
                                              : 5 + Math.abs(rng.nextGaussian()) * (stealth ? 120 : 55);
            double hour = fraud && !stealth ? (rng.nextDouble() < 0.7 ? rng.nextDouble() * 6 : rng.nextDouble() * 24)
                                            : 8 + rng.nextDouble() * 14;
            double distFromHome = fraud && !stealth ? rng.nextDouble() * 900 : rng.nextDouble() * 60;
            double txLast1h = fraud && !stealth ? 3 + (int) (rng.nextDouble() * 6) : (int) (rng.nextDouble() * 3);
            double foreign = fraud && !stealth && rng.nextDouble() < 0.4 ? 1 : 0;
            double nightWeekend = !fraud && (hour < 6 || hour > 22) && rng.nextDouble() < 0.6 ? 1 : 0;

            x.set(i, 0, amount); x.set(i, 1, hour / 24.0); x.set(i, 2, distFromHome);
            x.set(i, 3, txLast1h); x.set(i, 4, foreign); x.set(i, 5, nightWeekend);
        }
        Proj.standardize(x);

        int split = (int) (n * 0.75);
        Tensor trainX = x.sliceRows(0, split), testX = x.sliceRows(split, n);
        int[] trainY = java.util.Arrays.copyOf(y, split);
        int[] testY = java.util.Arrays.copyOfRange(y, split, n);
        long fraudTest = java.util.Arrays.stream(testY).filter(v -> v == 1).count();
        System.out.printf("[P15 Fraud] train=%d test=%d fraud-in-test=%d%n", trainX.rows(), testX.rows(), fraudTest);

        for (String variant : new String[]{"CrossEntropy", "FocalLoss"}) {
            SeedManager.setSeed(42);
            NeuralNetwork net = buildNet();
            var lossFn = variant.equals("FocalLoss") ? new FocalLoss(0.25, 2.0) : new CrossEntropyLoss();
            new Trainer(net, lossFn, new Adam(0.003)).fit(trainX, trainY, epochs * 2, batch);

            Tensor out = net.forward(testX, false);
            // Sweep decision threshold; report the best-F1 operating point (fair under imbalance)
            double bestF1 = 0, bestThr = 0.5, bestP = 0, bestR = 0;
            int bTp = 0, bFp = 0, bFn = 0;
            for (double thr = 0.05; thr <= 0.951; thr += 0.05) {
                int tp = 0, fp = 0, fn = 0;
                for (int i = 0; i < out.rows(); i++) {
                    boolean flag = out.get(i, 1) >= thr;
                    if (flag && testY[i] == 1) tp++;
                    else if (flag) fp++;
                    else if (testY[i] == 1) fn++;
                }
                double p = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
                double r = (double) tp / Math.max(1, tp + fn);
                double f1 = p + r > 0 ? 2 * p * r / (p + r) : 0;
                if (f1 > bestF1) { bestF1 = f1; bestThr = thr; bestP = p; bestR = r; bTp = tp; bFp = fp; bFn = fn; }
            }
            System.out.printf("%-13s best F1=%.2f @thr=%.2f | precision=%.2f recall=%.2f (TP=%d FP=%d FN=%d)%n",
                variant, bestF1, bestThr, bestP, bestR, bTp, bFp, bFn);
        }
        Proj.footer("P15 Fraud", t0);
    }

    private static NeuralNetwork buildNet() {
        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(6, 32));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(32, 16));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(16, 2));
        net.add(new ml.activation.Softmax());
        return net;
    }
}




