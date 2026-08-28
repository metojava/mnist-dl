package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class LogSigmoid extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> {
            if (x >= 0) {
                return -Math.log(1.0 + Math.exp(-x));
            } else {
                return x - Math.log(1.0 + Math.exp(x));
            }
        });
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++) {
            double x = lastInput.data()[i];
            double sigmoid = 1.0 / (1.0 + Math.exp(-x));
            result.data()[i] = gradient.data()[i] * (1.0 - sigmoid);
        }
        return result;
    }
}
