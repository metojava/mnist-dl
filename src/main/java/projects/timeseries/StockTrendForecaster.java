package projects.timeseries;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import ml.layer.LSTMLayer;
import projects.common.Proj;

import java.util.Random;

/**
 * P17 - Stock trend prediction (up/down next day) on a synthetic geometric-brownian-motion
 * price series with regime shifts. Sliding windows of normalized returns feed an LSTM.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.timeseries.StockTrendForecaster" "-Dexec.args=10 32"
 */
public class StockTrendForecaster {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 10), batch = Proj.batch(args, 32);
        long t0 = System.currentTimeMillis();

        double[] price = gbm(1200, new Random(42));
        int T = 20;

        // Features: last T daily log-returns; target: sign of next return
        int n = price.length - T - 1;
        Tensor x = new Tensor(n, T);
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t <= T; t++) {
                double ret = Math.log(price[i + t] / price[i + t > 0 ? i + t - 1 : 0]);
                if (t == T) y[i] = ret >= 0 ? 1 : 0;
                else x.set(i, t, ret * 100);   // in %
            }
        }
        Proj.shuffle(x, y, new Random(7));

        int split = (int) (n * 0.75);
        long ups = java.util.Arrays.stream(y).filter(v -> v == 1).count();
        System.out.printf("[P17 Stock] windows=%d up-ratio=%.2f%n", n, (double) ups / n);

        NeuralNetwork net = new NeuralNetwork();
        net.add(new LSTMLayer(1, 48));          // univariate sequence
        net.add(new ml.layer.DenseLayer(48, 16));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(16, 2));
        net.add(new ml.activation.Softmax());

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.005))
            .fit(x.sliceRows(0, split), java.util.Arrays.copyOf(y, split), epochs, batch);

        double accTrain = acc(net, x.sliceRows(0, split), java.util.Arrays.copyOf(y, split));
        double accTest = acc(net, x.sliceRows(split, n), java.util.Arrays.copyOfRange(y, split, n));
        System.out.printf("Majority baseline: %.1f%%%n", 100 * Math.max(ups, n - ups) / (double) n);
        System.out.printf("LSTM train acc: %.1f%% | test acc: %.1f%%%n", 100 * accTrain, 100 * accTest);
        Proj.footer("P17 Stock", t0);
    }

    /** Geometric brownian motion with occasional regime changes. */
    static double[] gbm(int steps, Random rng) {
        double[] p = new double[steps];
        p[0] = 100;
        double drift = 0.0003;
        for (int t = 1; t < steps; t++) {
            if (rng.nextDouble() < 0.01) drift = -drift * 2;   // regime shift
            double shock = rng.nextGaussian() * 0.02;
            p[t] = Math.max(1, p[t - 1] * Math.exp(drift + shock));
        }
        return p;
    }

    static double acc(NeuralNetwork net, Tensor x, int[] y) {
        Tensor out = net.forward(x, false);
        int c = 0;
        for (int i = 0; i < x.rows(); i++) if (out.argmaxRow(i) == y[i]) c++;
        return (double) c / x.rows();
    }
}
