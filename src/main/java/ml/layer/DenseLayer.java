package ml.layer;

import ml.core.Tensor;
import ml.core.ThreadPoolManager;
import ml.exception.ShapeException;

import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;

public class DenseLayer implements Layer {
    private final int inSize;
    private final int outSize;
    private final Tensor weights;
    private final Tensor bias;
    private volatile Tensor weightsGrad;
    private volatile Tensor biasGrad;
    private volatile Tensor lastInput;
    private final ReentrantLock gradLock = new ReentrantLock();
    private static final int PARALLEL_THRESHOLD = 8192;
    private Tensor cachedTranspose;

    public DenseLayer(int inputSize, int outputSize) {
        this.inSize = inputSize;
        this.outSize = outputSize;
        this.weights = Tensor.he(outputSize, inputSize);
        this.bias = Tensor.zeros(1, outputSize);
        this.cachedTranspose = this.weights.transpose();
    }

    @Deprecated
    public DenseLayer(int inputSize, int outputSize, java.util.Random rng) {
        this.inSize = inputSize;
        this.outSize = outputSize;
        this.weights = Tensor.he(outputSize, inputSize, rng);
        this.bias = Tensor.zeros(1, outputSize);
        this.cachedTranspose = this.weights.transpose();
    }

    public void invalidateTranspose() {
        this.cachedTranspose = null;
    }

    private Tensor getCachedTranspose() {
        if (cachedTranspose == null) {
            cachedTranspose = weights.transpose();
        }
        return cachedTranspose;
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        if (input.cols() != inSize)
            throw new ShapeException("DenseLayer forward: expected input cols=" + inSize + ", got " + input.cols());
        this.lastInput = input;

        Tensor output = Tensor.matmul(input, getCachedTranspose());

        // Parallel bias addition via CompletableFuture chunks
        int total = output.rows() * outSize;
        if (total <= PARALLEL_THRESHOLD) {
            addBiasSequential(output.data(), bias.data(), outSize, output.rows());
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, output.rows() / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(output.rows() + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, output.rows());
                futures[t] = CompletableFuture.runAsync(
                    () -> addBiasSequential(output.data(), bias.data(), outSize, from, to), pool);
            }
            CompletableFuture.allOf(futures).join();
        }
        return output;
    }

    private void addBiasSequential(double[] out, double[] b, int cols, int numRows) {
        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < cols; j++)
                out[i * cols + j] += b[j];
    }

    private void addBiasSequential(double[] out, double[] b, int cols, int rowFrom, int rowTo) {
        for (int i = rowFrom; i < rowTo; i++)
            for (int j = 0; j < cols; j++)
                out[i * cols + j] += b[j];
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int batchSize = gradient.rows();
        int totalWeightGrad = outSize * inSize;

        Tensor wGrad;
        Tensor bGrad;

        if ((long) batchSize * inSize * outSize <= PARALLEL_THRESHOLD) {
            wGrad = Tensor.matmul(gradient.transpose(), lastInput);
            bGrad = gradient.sumCols();
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.parallelism();
            int chunk = Math.max(1, batchSize / threads);
            int numTasks = (batchSize + chunk - 1) / chunk;

            double[][] partialDW = new double[numTasks][totalWeightGrad];
            double[][] partialDB = new double[numTasks][outSize];
            CompletableFuture<?>[] futures = new CompletableFuture[numTasks];

            for (int t = 0; t < numTasks; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, batchSize);
                final int taskIdx = t;
                futures[t] = CompletableFuture.runAsync(() -> {
                    double[] gradData = gradient.data();
                    double[] inData = lastInput.data();
                    double[] pdw = partialDW[taskIdx];
                    double[] pdb = partialDB[taskIdx];
                    for (int b = from; b < to; b++) {
                        int gradOffset = b * outSize;
                        int inOffset = b * inSize;
                        for (int oi = 0; oi < outSize; oi++) {
                            double gv = gradData[gradOffset + oi];
                            pdb[oi] += gv;
                            int pdwOffset = oi * inSize;
                            for (int ii = 0; ii < inSize; ii++) {
                                pdw[pdwOffset + ii] += gv * inData[inOffset + ii];
                            }
                        }
                    }
                }, pool);
            }
            CompletableFuture.allOf(futures).join();

            wGrad = Tensor.zeros(outSize, inSize);
            bGrad = Tensor.zeros(1, outSize);
            double[] wd = wGrad.data();
            double[] bd = bGrad.data();
            for (int t = 0; t < numTasks; t++) {
                for (int i = 0; i < totalWeightGrad; i++) wd[i] += partialDW[t][i];
                for (int i = 0; i < outSize; i++) bd[i] += partialDB[t][i];
            }
        }

        gradLock.lock();
        try {
            this.weightsGrad = wGrad;
            this.biasGrad = bGrad;
        } finally {
            gradLock.unlock();
        }

        // dX = dZ * W^T — parallel matmul
        return Tensor.matmul(gradient, weights);
    }

    @Override
    public void update(double learningRate) {
        gradLock.lock();
        try {
            if (weightsGrad == null) return;
            double[] wData = weights.data();
            double[] gData = weightsGrad.data();
            double[] bData = bias.data();
            double[] bgData = biasGrad.data();
            int wLen = wData.length;

            // Parallel weight update via ExecutorService
            if (wLen <= PARALLEL_THRESHOLD) {
                for (int i = 0; i < wLen; i++) wData[i] -= learningRate * gData[i];
                for (int i = 0; i < outSize; i++) bData[i] -= learningRate * bgData[i];
            } else {
                ExecutorService pool = ThreadPoolManager.compute();
                int threads = ThreadPoolManager.parallelism();
                int chunk = Math.max(1, wLen / threads);
                CompletableFuture<?>[] futures = new CompletableFuture[(wLen + chunk - 1) / chunk];
                for (int t = 0; t < futures.length; t++) {
                    int from = t * chunk;
                    int to = Math.min(from + chunk, wLen);
                    futures[t] = CompletableFuture.runAsync(() -> {
                        for (int i = from; i < to; i++) wData[i] -= learningRate * gData[i];
                    }, pool);
                }
                CompletableFuture.allOf(futures).join();
                for (int i = 0; i < outSize; i++) bData[i] -= learningRate * bgData[i];
            }
            invalidateTranspose();
        } finally {
            gradLock.unlock();
        }
    }

    @Override
    public int inputSize() { return inSize; }
    @Override
    public int outputSize() { return outSize; }
    @Override
    public int parameterCount() { return inSize * outSize + outSize; }

    public Tensor weights() { return weights; }
    public Tensor bias() { return bias; }
    public Tensor weightsGradient() { return weightsGrad; }
    public Tensor biasGradient() { return biasGrad; }
}
