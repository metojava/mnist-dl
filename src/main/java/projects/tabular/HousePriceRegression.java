package projects.tabular;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.optimizer.Adam;
import projects.common.Proj;

import java.util.Random;

/**
 * P14 - House price regression (synthetic housing market).
 * Features: size(m2), rooms, age, garage(0/1), distance-to-center(km). Target: price (EUR).
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.tabular.HousePriceRegression" "-Dexec.args=150 32"
 */
public class HousePriceRegression {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 150), batch = Proj.batch(args, 32);
        long t0 = System.currentTimeMillis();

        int n = 1200;
        Random rng = new Random(42);
        double[][] rawX = new double[n][5];
        double[] rawY = new double[n];
        for (int i = 0; i < n; i++) {
            double size = 60 + rng.nextDouble() * 190;
            double rooms = 1 + (int) (rng.nextDouble() * 6);
            double age = rng.nextDouble() * 90;
            double garage = rng.nextDouble() < 0.45 ? 1 : 0;
            double dist = 0.5 + rng.nextDouble() * 24;
            rawX[i] = new double[]{size, rooms, age, garage, dist};
            rawY[i] = 42_000 * size / 100 + 18_000 * rooms - 900 * age
                + 25_000 * garage - 3_500 * dist + 12_000 * rng.nextGaussian();
        }

        Stats xs = stats(rawX, 5);
        Stats ys = stat1(rawY);

        Tensor x = standardize(rawX, xs);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) y[i] = (rawY[i] - ys.mean()[0]) * ys.inv()[0];

        int split = (int) (n * 0.8);
        Tensor trainX = x.sliceRows(0, split), testX = x.sliceRows(split, n);
        double[] trainY = java.util.Arrays.copyOf(y, split);

        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(5, 64));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(64, 32));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(32, 1));
        System.out.printf("[P14 HousePrice] samples=%d%n", n);

        for (int ep = 1; ep <= epochs; ep++) {
            int from = 0;
            double loss = 0;
            while (from < split) {
                int to = Math.min(from + batch, split);
                Tensor xb = trainX.sliceRows(from, to);
                Tensor pred = net.forward(xb, true);
                Tensor target = Tensor.fromFlat(java.util.Arrays.copyOfRange(trainY, from, to), to - from, 1);
                loss += mse(pred, target);
                Tensor grad = pred.subtract(target).multiply(2.0 / (to - from));
                net.backward(grad);
                net.update(0.01);
                from = to;
            }
            if (ep % 30 == 0 || ep == 1) System.out.printf("epoch %d: MSE=%.5f%n", ep, loss / Math.ceil((double) split / batch));
        }

        // Test RMSE in EUR
        Tensor pred = net.forward(testX, false);
        double se = 0;
        for (int i = 0; i < testX.rows(); i++) {
            double p = pred.data()[i] * ys.inv()[0] + ys.mean()[0];
            double d = p - rawY[split + i];
            se += d * d;
        }
        double rmse = Math.sqrt(se / testX.rows());
        System.out.printf("Test RMSE: %.0f EUR (%.1f%% of mean %.0f EUR)%n", rmse, 100 * rmse / ys.mean()[0], ys.mean()[0]);

        double[] demo = {110, 4, 15, 1, 6};
        double[] std = new double[5];
        for (int j = 0; j < 5; j++) std[j] = (demo[j] - xs.mean[j]) * xs.inv[j];
        double est = net.forward(Tensor.fromFlat(std, 1, 5), false).data()[0] * ys.inv()[0] + ys.mean()[0];
        System.out.printf("Demo house %s -> ~%,.0f EUR%n", java.util.Arrays.toString(demo), est);
        Proj.footer("P14 HousePrice", t0);
    }

    record Stats(double[] mean, double[] inv) {}

    static Stats stats(double[][] m, int cols) {
        double[] mean = new double[cols], inv = new double[cols];
        for (int j = 0; j < cols; j++) {
            double s = 0;
            for (double[] row : m) s += row[j];
            mean[j] = s / m.length;
            double v = 0;
            for (double[] row : m) { double d = row[j] - mean[j]; v += d * d; }
            inv[j] = 1.0 / Math.sqrt(v / Math.max(1, m.length - 1) + 1e-9);
        }
        return new Stats(mean, inv);
    }

    static Stats stat1(double[] a) {
        double m = java.util.Arrays.stream(a).average().orElse(0);
        double v = 0;
        for (double val : a) v += (val - m) * (val - m);
        return new Stats(new double[]{m}, new double[]{1.0 / Math.sqrt(v / Math.max(1, a.length - 1) + 1e-9)});
    }

    static Tensor standardize(double[][] m, Stats s) {
        Tensor t = new Tensor(m.length, m[0].length);
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                t.set(i, j, (m[i][j] - s.mean[j]) * s.inv[j]);
        return t;
    }

    static double mse(Tensor a, Tensor b) {
        double s = 0;
        for (int i = 0; i < a.size(); i++) { double d = a.data()[i] - b.data()[i]; s += d * d; }
        return s / a.size();
    }
}


