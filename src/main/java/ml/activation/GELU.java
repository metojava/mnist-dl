package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class GELU extends ActivationLayer {
    private static final double SQRT_2_PI = Math.sqrt(2.0 / Math.PI);
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
            double cdf = 0.5 * (1.0 + Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x)));
            out[i] = (float) (x * cdf);
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
            double tanhVal = Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x));
            double sech2 = 1.0 - tanhVal * tanhVal;
            double dCdf = 0.5 * SQRT_2_PI * sech2 * (1.0 + 3.0 * 0.044715 * x * x);
            double dGelu = 0.5 * (1.0 + tanhVal) + x * dCdf;
            r[i] = (float) (g[i] * dGelu);
        }
        return gradientBuffer;
    }
}
