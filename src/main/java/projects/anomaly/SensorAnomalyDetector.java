package projects.anomaly;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.optimizer.Adam;
import projects.common.Proj;

import java.util.Random;

/**
 * P20 - Sensor anomaly detection with an autoencoder.
 * Trained only on normal 3-sensor telemetry; anomalies (spikes, stuck-at values)
 * are flagged when reconstruction error exceeds the 95th percentile of validation error.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.anomaly.SensorAnomalyDetector" "-Dexec.args=200 64"
 */
public class SensorAnomalyDetector {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 200), batch = Proj.batch(args, 64);
        long t0 = System.currentTimeMillis();

        Random rng = new Random(42);
        int trainN = 3000, testN = 1200;

        // ---- Normal data: smooth sinusoid + drift + noise ----
        Tensor trainX = new Tensor(trainN, 3);
        for (int i = 0; i < trainN; i++)
            for (int s = 0; s < 3; s++) {
                double v = 10 * Math.sin(2 * Math.PI * i / 180.0 + s)
                    + 2.5 * Math.sin(2 * Math.PI * i / 24.0 + 1.3 * s)
                    + rng.nextGaussian() * 0.4;
                trainX.set(i, s, (v + 12) / 24);   // -> ~[0,1]
            }

        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(3, 16));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(16, 6));    // bottleneck
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(6, 16));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(16, 3));
        net.add(new ml.activation.Sigmoid());
        System.out.printf("[P20 Anomaly] params=%d%n", net.totalParameters());

        double lr = 0.008;
        for (int ep = 1; ep <= epochs; ep++) {
            double loss = 0; int cnt = 0;
            int from = 0;
            while (from < trainN) {
                int to = Math.min(from + batch, trainN);
                Tensor xb = trainX.sliceRows(from, to);
                Tensor pred = net.forward(xb, true);
                loss += mse(pred, xb);
                Tensor grad = pred.subtract(xb).multiply(2.0 / (to - from));
                net.backward(grad);
                net.update(lr);
                from = to; cnt++;
            }
            if (ep == 1 || ep % 50 == 0) System.out.printf("epoch %d: recon MSE=%.5f%n", ep, loss / cnt);
        }

        // Threshold = p95 of training reconstruction errors
        double[] errs = new double[trainN];
        Tensor recTrain = net.forward(trainX, false);
        for (int i = 0; i < trainN; i++) {
            double e = 0;
            for (int s = 0; s < 3; s++) { double d = recTrain.get(i, s) - trainX.get(i, s); e += d * d; }
            errs[i] = Math.sqrt(e);
        }
        java.util.Arrays.sort(errs);
        double threshold = errs[(int) (errs.length * 0.99)];
        System.out.printf("anomaly threshold (p99): %.4f%n", threshold);

        // ---- Test stream: mostly normal, inject anomalies every ~40 steps ----
        Tensor testX = new Tensor(testN, 3);
        boolean[] isAnomaly = new boolean[testN];
        for (int i = 0; i < testN; i++) {
            boolean anomalous = i % 40 == 17;
            isAnomaly[i] = anomalous;
            for (int s = 0; s < 3; s++) {
                double v = 10 * Math.sin(2 * Math.PI * i / 180.0 + s)
                    + 2.5 * Math.sin(2 * Math.PI * i / 24.0 + 1.3 * s)
                    + rng.nextGaussian() * 0.4;
                if (anomalous) {
                    int mode = (i + s) % 3;
                    if (mode == 0) v += rng.nextBoolean() ? 11 : -11;          // spike
                    else if (mode == 1) v = tempAt(i - 30, s) ;                // stuck-at
                    else v += rng.nextGaussian() * 6;                          // heavy noise
                }
                testX.set(i, s, (v + 12) / 24);
            }
        }

        Tensor recTest = net.forward(testX, false);
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < testN; i++) {
            double e = 0;
            for (int s = 0; s < 3; s++) { double d = recTest.get(i, s) - testX.get(i, s); e += d * d; }
            boolean flag = Math.sqrt(e) > threshold;
            if (flag && isAnomaly[i]) tp++;
            else if (flag) fp++;
            else if (isAnomaly[i]) fn++;
        }
        double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
        double recall = (double) tp / Math.max(1, tp + fn);
        System.out.printf("Detected anomalies: TP=%d FP=%d FN=%d | precision=%.2f recall=%.2f F1=%.2f%n",
            tp, fp, fn, precision, recall,
            precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0);
        Proj.footer("P20 Anomaly", t0);
    }

    static double[] stuckCache = new double[3];

    static double tempAt(int i, int s) {
        if (i < 0) i = 0;
        return 10 * Math.sin(2 * Math.PI * i / 180.0 + s) + 2.5 * Math.sin(2 * Math.PI * i / 24.0 + 1.3 * s);
    }

    static double mse(Tensor a, Tensor b) {
        double s = 0;
        for (int i = 0; i < a.size(); i++) { double d = a.data()[i] - b.data()[i]; s += d * d; }
        return s / a.size();
    }
}

