package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Sigmoid extends ActivationLayer {
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
            out[i] = (float) (1.0 / (1.0 + Math.exp(-x)));
        }
        return outputBuffer;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int rows = lastOutput.rows();
        int cols = lastOutput.cols();
        if (gradientBuffer == null || gradientBuffer.rows() != rows || gradientBuffer.cols() != cols)
            gradientBuffer = new Tensor(rows, cols);

        float[] o = lastOutput.data();
        float[] g = gradient.data();
        float[] r = gradientBuffer.data();
        for (int i = 0; i < r.length; i++) {
            double s = o[i];
            r[i] = (float) (g[i] * s * (1 - s));
        }
        return gradientBuffer;
    }
}
