package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class ELU extends ActivationLayer {
    private final double alpha;
    private Tensor outputBuffer;
    private Tensor gradientBuffer;

    public ELU(double alpha) {
        this.alpha = alpha;
    }

    public ELU() {
        this(1.0);
    }

    public double alpha() { return alpha; }

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
            out[i] = (float) (x > 0 ? x : alpha * (Math.exp(x) - 1));
        }
        return outputBuffer;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int rows = lastInput.rows();
        int cols = lastInput.cols();
        if (gradientBuffer == null || gradientBuffer.rows() != rows || gradientBuffer.cols() != cols)
            gradientBuffer = new Tensor(rows, cols);

        float[] in = lastInput.data();
        float[] g = gradient.data();
        float[] r = gradientBuffer.data();
        for (int i = 0; i < r.length; i++) {
            double x = in[i];
            r[i] = (float) (x > 0 ? g[i] : g[i] * alpha * Math.exp(x));
        }
        return gradientBuffer;
    }
}
