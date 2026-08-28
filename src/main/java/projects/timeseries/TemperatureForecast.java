package projects.timeseries;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.loss.MSELoss;
import ml.optimizer.Adam;
import ml.layer.LSTMLayer;
import projects.common.Proj;

import java.util.Random;

/**
 * P18 - Air temperature forecast (next-hour) on a synthetic seasonal climate series.
 * Daily + annual cycles with noise; LSTM predicts the next temperature from the
 * previous 24 hours. Reports MAE vs a naive "last value" baseline.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.timeseries.TemperatureForecast" "-Dexec.args=12 32"
 */
public class TemperatureForecast {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 12), batch = Proj.batch(args, 32);
        long t0 = System.currentTimeMillis();

        int hours = 3000;
        double[] temp = new double[hours];
        Random rng = new Random(42);
        for (int h = 0; h < hours; h++) {
            double daily = 6 * Math.sin(2 * Math.PI * (h % 24) / 24 - Math.PI / 2);
            double season = 9 * Math.sin(2 * Math.PI * h / (24 * 365));
            double noise = rng.nextGaussian() * 1.2;
            temp[h] = 14 + season + daily + noise;
        }

        int T = 24;
        int n = hours - T - 1;

        // Normalization stats
        double mean = java.util.Arrays.stream(temp).average().orElse(0);
        double std = Math.sqrt(java.util.Arrays.stream(temp).map(v -> (v - mean) * (v - mean)).sum() / (hours - 1));

        Tensor x = new Tensor(n, T);
        double[] target = new double[n];
        double[] lastValue = new double[n];   // naive baseline
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < T; t++) x.set(i, t, (temp[i + t] - mean) / std);
            target[i] = (temp[i + T] - mean) / std;
            lastValue[i] = temp[i + T - 1];
        }

        int split = (int) (n * 0.8);
        Tensor trainX = x.sliceRows(0, split);
        NeuralNetwork net = new NeuralNetwork();
        net.add(new LSTMLayer(1, 64));
        net.add(new ml.layer.DenseLayer(64, 32));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(32, 1));
        
        System.out.printf("[P18 Temperature] windows=%d lookback=%dh%n", n, T);

        for (int ep = 0; ep < epochs; ep++) {
            double loss = 0; int cnt = 0;
            for (int from = 0; from < split; from += batch) {
                int to = Math.min(from + batch, split);
                Tensor xb = trainX.sliceRows(from, to);
                int bs = to - from;
                Tensor pred = net.forward(xb, true);
                Tensor grad = new Tensor(bs, 1);
                double s = 0;
                for (int i = 0; i < bs; i++) {
                    double d2 = pred.data()[i] - target[from + i];
                    grad.set(i, 0, d2);
                    s += d2 * d2;
                }
                loss += s / bs;
                grad.multiplyInPlace(2.0 / bs);
                net.backward(grad);
                net.update(0.005);
                cnt++;
            }
            if (ep == 0 || (ep + 1) % 4 == 0)
                System.out.printf("epoch %d: MSE=%.5f%n", ep + 1, loss / Math.max(1, cnt));
        }

        // Evaluation
        Tensor testX = x.sliceRows(split, n);
        Tensor pred = net.forward(testX, false);
        double maeModel = 0, maeNaive = 0;
        for (int i = 0; i < testX.rows(); i++) {
            double truth = temp[split + i + T];
            maeModel += Math.abs(pred.data()[i] * std + mean - truth);
            maeNaive += Math.abs(lastValue[split + i] - truth);
        }
        System.out.printf("MAE model : %.3f C%n", maeModel / testX.rows());
        System.out.printf("MAE naive : %.3f C (last-value persistence)%n", maeNaive / testX.rows());
        Proj.footer("P18 Temperature", t0);
    }

    private static void d(double v, Tensor grad, int row) { grad.set(row, 0, v); }
}

