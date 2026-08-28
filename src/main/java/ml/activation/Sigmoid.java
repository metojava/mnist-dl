package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Sigmoid extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> 1.0 / (1.0 + Math.exp(-x)));
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastOutput.rows(), lastOutput.cols());
        for (int i = 0; i < lastOutput.size(); i++) {
            double s = lastOutput.data()[i];
            result.data()[i] = gradient.data()[i] * s * (1 - s);
        }
        return result;
    }
}
