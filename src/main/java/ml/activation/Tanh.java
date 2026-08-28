package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Tanh extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(Math::tanh);
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastOutput.rows(), lastOutput.cols());
        for (int i = 0; i < lastOutput.size(); i++) {
            double t = lastOutput.data()[i];
            result.data()[i] = gradient.data()[i] * (1 - t * t);
        }
        return result;
    }
}
