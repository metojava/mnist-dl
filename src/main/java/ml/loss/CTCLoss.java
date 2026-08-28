package ml.loss;

import ml.core.Tensor;

public class CTCLoss implements Loss {
    private final double blank;
    private final boolean zeroInfinity;

    public CTCLoss(double blank, boolean zeroInfinity) {
        this.blank = blank;
        this.zeroInfinity = zeroInfinity;
    }

    public CTCLoss() {
        this(0, true);
    }

    public CTCLoss(double blank) {
        this(blank, true);
    }

    public double blank() { return blank; }

    @Override
    public double forward(Tensor predictions, int[] labels) {
        int T = predictions.cols();
        int batchSize = labels.length;
        int C = predictions.rows() / batchSize > 0 ? predictions.rows() : T;
        return ctcLossForward(predictions, labels, T, batchSize);
    }

    private double ctcLossForward(Tensor logProbs, int[] targets, int T, int batchSize) {
        int S = targets.length;
        int U = 2 * S + 1;
        double totalLoss = 0;

        double[][] alpha = new double[T][U];
        for (double[] row : alpha) java.util.Arrays.fill(row, Double.NEGATIVE_INFINITY);
        alpha[0][0] = logProbs.get(0, (int) blank);
        if (U > 1) alpha[0][1] = logProbs.get(0, targets[0]);

        for (int t = 1; t < T; t++) {
            for (int u = 0; u < U; u++) {
                double val = alpha[t - 1][u];
                if (u > 0) val = logSumExp(val, alpha[t - 1][u - 1]);
                if (u > 1 && u % 2 == 1 && targets[u / 2] != targets[u / 2 - 1]) {
                    val = logSumExp(val, alpha[t - 1][u - 2]);
                } else if (u > 2 && u % 2 == 0) {
                    val = logSumExp(val, alpha[t - 1][u - 2]);
                }
                int emit = (u % 2 == 0) ? (int) blank : targets[u / 2];
                alpha[t][u] = val + logProbs.get(t, emit);
            }
        }

        double last = alpha[T - 1][U - 1];
        if (U > 1) last = logSumExp(last, alpha[T - 1][U - 2]);

        totalLoss = -last;
        if (zeroInfinity && (Double.isInfinite(totalLoss) || Double.isNaN(totalLoss))) {
            totalLoss = 0;
        }

        return totalLoss / batchSize;
    }

    @Override
    public Tensor backward(Tensor predictions, int[] labels) {
        int T = predictions.cols();
        int batchSize = labels.length;
        Tensor grad = Tensor.zeros(predictions.rows(), predictions.cols());

        int S = labels.length;
        int U = 2 * S + 1;

        double[][] alpha = new double[T][U];
        for (double[] row : alpha) java.util.Arrays.fill(row, Double.NEGATIVE_INFINITY);
        alpha[0][0] = predictions.get(0, (int) blank);
        if (U > 1) alpha[0][1] = predictions.get(0, labels[0]);

        for (int t = 1; t < T; t++) {
            for (int u = 0; u < U; u++) {
                double val = alpha[t - 1][u];
                if (u > 0) val = logSumExp(val, alpha[t - 1][u - 1]);
                if (u > 1 && u % 2 == 1 && labels[u / 2] != labels[u / 2 - 1]) {
                    val = logSumExp(val, alpha[t - 1][u - 2]);
                } else if (u > 2 && u % 2 == 0) {
                    val = logSumExp(val, alpha[t - 1][u - 2]);
                }
                int emit = (u % 2 == 0) ? (int) blank : labels[u / 2];
                alpha[t][u] = val + predictions.get(t, emit);
            }
        }

        double[][] beta = new double[T][U];
        for (double[] row : beta) java.util.Arrays.fill(row, Double.NEGATIVE_INFINITY);
        beta[T - 1][U - 1] = 0;
        if (U > 1) beta[T - 1][U - 2] = 0;

        for (int t = T - 2; t >= 0; t--) {
            for (int u = 0; u < U; u++) {
                double val = beta[t + 1][u];
                if (u < U - 1) val = logSumExp(val, beta[t + 1][u + 1]);
                if (u < U - 2 && u % 2 == 0 && labels[u / 2] != labels[u / 2 + 1]) {
                    val = logSumExp(val, beta[t + 1][u + 2]);
                } else if (u < U - 3 && u % 2 == 1) {
                    val = logSumExp(val, beta[t + 1][u + 2]);
                }
                int emit = (u % 2 == 0) ? (int) blank : labels[u / 2];
                beta[t][u] = val + predictions.get(t, emit);
            }
        }

        double logProb = alpha[T - 1][U - 1];
        if (U > 1) logProb = logSumExp(logProb, alpha[T - 1][U - 2]);

        for (int t = 0; t < T; t++) {
            double[] classSum = new double[predictions.cols()];
            for (int u = 0; u < U; u++) {
                int emit = (u % 2 == 0) ? (int) blank : labels[u / 2];
                classSum[emit] = logSumExp(classSum[emit], alpha[t][u] + beta[t][u]);
            }
            for (int c = 0; c < predictions.cols(); c++) {
                double gradVal = Math.exp(classSum[c] - logProb - predictions.get(t, c));
                grad.data()[c * T + t] = -gradVal;
            }
        }

        if (zeroInfinity) {
            for (int i = 0; i < grad.size(); i++) {
                if (Double.isInfinite(grad.data()[i]) || Double.isNaN(grad.data()[i])) {
                    grad.data()[i] = 0;
                }
            }
        }

        return grad;
    }

    private double logSumExp(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        if (a > b) return a + Math.log(1.0 + Math.exp(b - a));
        return b + Math.log(1.0 + Math.exp(a - b));
    }
}
