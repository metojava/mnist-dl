package ml.core;

import java.util.Random;

public final class Initializers {
    private Initializers() {}

    public static double[] heWeights(int inputSize, int outputSize, Random rng) {
        double std = Math.sqrt(2.0 / inputSize);
        double[] w = new double[inputSize * outputSize];
        for (int i = 0; i < w.length; i++) w[i] = rng.nextGaussian() * std;
        return w;
    }

    public static double[] xavierWeights(int inputSize, int outputSize, Random rng) {
        double std = Math.sqrt(2.0 / (inputSize + outputSize));
        double[] w = new double[inputSize * outputSize];
        for (int i = 0; i < w.length; i++) w[i] = rng.nextGaussian() * std;
        return w;
    }

    public static double[] lecunWeights(int inputSize, int outputSize, Random rng) {
        double std = Math.sqrt(1.0 / inputSize);
        double[] w = new double[inputSize * outputSize];
        for (int i = 0; i < w.length; i++) w[i] = rng.nextGaussian() * std;
        return w;
    }

    public static double[] zeros(int size) {
        return new double[size];
    }

    public static double[] ones(int size) {
        double[] arr = new double[size];
        java.util.Arrays.fill(arr, 1.0);
        return arr;
    }

    public static Tensor heTensor(int rows, int cols, Random rng) {
        return Tensor.he(rows, cols, rng);
    }

    public static Tensor xavierTensor(int rows, int cols, Random rng) {
        return Tensor.xavier(rows, cols, rng);
    }

    public static double[] glorotUniform(int inputSize, int outputSize, Random rng) {
        double limit = Math.sqrt(6.0 / (inputSize + outputSize));
        double[] w = new double[inputSize * outputSize];
        for (int i = 0; i < w.length; i++) w[i] = (rng.nextDouble() * 2 - 1) * limit;
        return w;
    }

    public static double[] glorotNormal(int inputSize, int outputSize, Random rng) {
        double std = Math.sqrt(2.0 / (inputSize + outputSize));
        double[] w = new double[inputSize * outputSize];
        for (int i = 0; i < w.length; i++) w[i] = rng.nextGaussian() * std;
        return w;
    }

    public static double[] truncatedNormal(int inputSize, int outputSize, double stddev, Random rng) {
        double[] w = new double[inputSize * outputSize];
        for (int i = 0; i < w.length; i++) {
            double val;
            do { val = rng.nextGaussian() * stddev; } while (Math.abs(val) > 2 * stddev);
            w[i] = val;
        }
        return w;
    }

    public static double[] truncatedNormal(int inputSize, int outputSize, Random rng) {
        return truncatedNormal(inputSize, outputSize, 0.05, rng);
    }

    public static double[] orthogonal(int rows, int cols, Random rng) {
        double[][] h = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                h[i][j] = rng.nextGaussian();

        int minDim = Math.min(rows, cols);
        double[][] u = new double[rows][minDim];
        double[][] v = new double[minDim][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < minDim; j++)
                u[i][j] = h[i][j];

        for (int iter = 0; iter < 100; iter++) {
            double[][] utU = new double[minDim][minDim];
            for (int i = 0; i < minDim; i++) {
                for (int j = 0; j < minDim; j++) {
                    double sum = 0;
                    for (int k = 0; k < rows; k++) sum += u[k][i] * u[k][j];
                    utU[i][j] = sum;
                }
            }
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < minDim; j++) {
                    double sum = 0;
                    for (int k = 0; k < minDim; k++) sum += u[i][k] * utU[k][j];
                    u[i][j] = h[i][j] - sum;
                }
        }

        double[] w = new double[rows * cols];
        double scale = Math.sqrt((double) cols / rows);
        for (int i = 0; i < rows; i++) {
            double norm = 0;
            for (int j = 0; j < minDim; j++) norm += u[i][j] * u[i][j];
            norm = Math.sqrt(norm) + 1e-12;
            for (int j = 0; j < minDim; j++) u[i][j] = u[i][j] / norm * scale;
        }

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                w[i * cols + j] = j < minDim ? u[i][j] : 0;

        return w;
    }

    public static Tensor glorotUniformTensor(int rows, int cols, Random rng) {
        double limit = Math.sqrt(6.0 / (rows + cols));
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) data[i] = (rng.nextDouble() * 2 - 1) * limit;
        return Tensor.fromFlat(data, rows, cols);
    }

    public static Tensor glorotNormalTensor(int rows, int cols, Random rng) {
        double std = Math.sqrt(2.0 / (rows + cols));
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) data[i] = rng.nextGaussian() * std;
        return Tensor.fromFlat(data, rows, cols);
    }

    public static Tensor orthogonalTensor(int rows, int cols, Random rng) {
        double[] data = orthogonal(rows, cols, rng);
        return Tensor.fromFlat(data, rows, cols);
    }
}
