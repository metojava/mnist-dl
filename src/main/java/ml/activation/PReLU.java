package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class PReLU extends ActivationLayer {
    private final double alphaInit;
    private double alpha;
    private double alphaGrad;
    private Tensor outputBuffer;
    private Tensor gradientBuffer;

    public PReLU(double alphaInit) {
        this.alphaInit = alphaInit;
        this.alpha = alphaInit;
    }

    public PReLU() {
        this(0.25);
    }

    public double alpha() { return alpha; }

    public void setAlpha(double alpha) { this.alpha = alpha; }

    @Override
    protected Tensor computeForward(Tensor input) {
        int rows = input.rows();
        int cols = input.cols();
        if (outputBuffer == null || outputBuffer.rows() != rows || outputBuffer.cols() != cols)
            outputBuffer = new Tensor(rows, cols);

        float[] in = input.data();
        float[] out = outputBuffer.data();
        for (int i = 0; i < out.length; i++) {
            double x = in[i];
            out[i] = (float) (x > 0 ? x : alpha * x);
        }
        return outputBuffer;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int rows = lastInput.rows();
        int cols = lastInput.cols();
        if (gradientBuffer == null || gradientBuffer.rows() != rows || gradientBuffer.cols() != cols)
            gradientBuffer = new Tensor(rows, cols);

        int len = lastInput.size();
        float[] in = lastInput.data();
        float[] g = gradient.data();
        float[] r = gradientBuffer.data();
        double gradSum = 0;
        for (int i = 0; i < len; i++) {
            double x = in[i];
            if (x > 0) {
                r[i] = g[i];
            } else {
                r[i] = (float) (alpha * g[i]);
                gradSum += g[i] * x;
            }
        }
        this.alphaGrad = gradSum / len;
        return gradientBuffer;
    }

    @Override
    public void update(double learningRate) {
        alpha -= learningRate * alphaGrad;
    }

    @Override
    public int parameterCount() { return 1; }
}
