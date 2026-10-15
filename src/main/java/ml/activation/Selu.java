package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Selu extends ActivationLayer {
    private static final double ALPHA = 1.6732632423543772848170429916717;
    private static final double SCALE = 1.0507009873554804934193349852946;
    private Tensor outputBuffer;
    private Tensor gradientBuffer;

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
            out[i] = (float) (x > 0 ? SCALE * x : SCALE * ALPHA * (Math.exp(x) - 1));
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
            r[i] = (float) (x > 0 ? g[i] * SCALE : g[i] * SCALE * ALPHA * Math.exp(x));
        }
        return gradientBuffer;
    }
}
