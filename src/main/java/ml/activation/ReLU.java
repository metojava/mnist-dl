package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class ReLU extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> Math.max(0, x));
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int len = lastInput.size();
        Tensor result = new Tensor(lastInput.rows(), lastInput.cols());
        double[] in = lastInput.data();
        double[] g = gradient.data();
        double[] r = result.data();
        for (int i = 0; i < len; i++)
            r[i] = in[i] > 0 ? g[i] : 0;
        return result;
    }
}
