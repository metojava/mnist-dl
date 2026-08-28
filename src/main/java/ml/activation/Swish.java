package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Swish extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> x / (1.0 + Math.exp(-x)));
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++) {
            double x = lastInput.data()[i];
            double sigmoid = 1.0 / (1.0 + Math.exp(-x));
            double dSwish = sigmoid + x * sigmoid * (1 - sigmoid);
            result.data()[i] = gradient.data()[i] * dSwish;
        }
        return result;
    }
}
