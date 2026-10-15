package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class HardSwish extends ActivationLayer {
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
            double relu6 = Math.min(Math.max(x + 3.0, 0.0), 6.0);
            out[i] = (float) (x * relu6 / 6.0);
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
            double dHardSwish;
            if (x <= -3.0) {
                dHardSwish = 0.0;
            } else if (x < 3.0) {
                dHardSwish = (2.0 * x + 3.0) / 6.0;
            } else {
                dHardSwish = 1.0;
            }
            r[i] = (float) (g[i] * dHardSwish);
        }
        return gradientBuffer;
    }
}
