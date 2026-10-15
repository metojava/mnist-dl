package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Softplus extends ActivationLayer {
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
            if (x > 20.0) {
                out[i] = (float) x;
            } else if (x < -20.0) {
                out[i] = 0.0f;
            } else {
                out[i] = (float) Math.log(1.0 + Math.exp(x));
            }
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
            double sigmoid;
            if (x > 20.0) {
                sigmoid = 1.0;
            } else if (x < -20.0) {
                sigmoid = 0.0;
            } else {
                sigmoid = 1.0 / (1.0 + Math.exp(-x));
            }
            r[i] = (float) (g[i] * sigmoid);
        }
        return gradientBuffer;
    }
}
