package ml.layer;

import ml.core.Tensor;
import ml.exception.ShapeException;

public class DenseLayer implements Layer {
    private final int inSize;
    private final int outSize;
    private final Tensor weights;
    private final Tensor bias;

    // Gradient / activation buffers are reused across batches so the hot
    // training loop produces no per-batch garbage.
    private Tensor weightsGrad;
    private Tensor biasGrad;
    private Tensor lastInput;
    private Tensor outputBuffer;
    private Tensor inputGradientBuffer;

    // W^T (inSize x outSize) used by forward; rebuilt only when weights change.
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
        if (cachedTranspose == null) cachedTranspose = weights.transpose();
        return cachedTranspose;
    }

    private static Tensor ensure(Tensor tensor, int rows, int cols) {
        if (tensor == null || tensor.rows() != rows || tensor.cols() != cols)
            return new Tensor(rows, cols);
        return tensor;
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        if (input.cols() != inSize)
            throw new ShapeException("DenseLayer forward: expected input cols=" + inSize + ", got " + input.cols());
        this.lastInput = input;

        outputBuffer = ensure(outputBuffer, input.rows(), outSize);
        Tensor.matmul(input, getCachedTranspose(), outputBuffer);

        float[] out = outputBuffer.data();
        float[] b = bias.data();
        int batchSize = input.rows();
        for (int i = 0; i < batchSize; i++) {
            int off = i * outSize;
            for (int j = 0; j < outSize; j++) out[off + j] += b[j];
        }
        return outputBuffer;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int batchSize = gradient.rows();

        // dW = X^T dZ  -> [outSize x inSize], same shape as weights.
        weightsGrad = ensure(weightsGrad, outSize, inSize);
        Tensor.matmulTransposedA(gradient, lastInput, weightsGrad);

        // db = sum(dZ) over the batch dimension -> [1 x outSize].
        biasGrad = ensure(biasGrad, 1, outSize);
        float[] bgd = biasGrad.data();
        float[] gd = gradient.data();
        java.util.Arrays.fill(bgd, 0.0f);
        for (int i = 0; i < batchSize; i++) {
            int off = i * outSize;
            for (int j = 0; j < outSize; j++) bgd[j] += gd[off + j];
        }

        // dX = dZ W  -> [batchSize x inSize]; no transpose needed.
        inputGradientBuffer = ensure(inputGradientBuffer, batchSize, inSize);
        Tensor.matmul(gradient, weights, inputGradientBuffer);
        return inputGradientBuffer;
    }

    @Override
    public void update(double learningRate) {
        if (weightsGrad == null) return;
        float[] wData = weights.data();
        float[] gData = weightsGrad.data();
        float[] bData = bias.data();
        float[] bgData = biasGrad.data();
        for (int i = 0; i < wData.length; i++) wData[i] -= learningRate * gData[i];
        for (int i = 0; i < bData.length; i++) bData[i] -= learningRate * bgData[i];
        invalidateTranspose();
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
