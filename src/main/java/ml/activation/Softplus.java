package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Softplus extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> {
            if (x > 20.0) return x;
            if (x < -20.0) return 0.0;
            return Math.log(1.0 + Math.exp(x));
        });
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++) {
            double x = lastInput.data()[i];
            double sigmoid;
            if (x > 20.0) {
                sigmoid = 1.0;
            } else if (x < -20.0) {
                sigmoid = 0.0;
            } else {
                sigmoid = 1.0 / (1.0 + Math.exp(-x));
            }
            result.data()[i] = gradient.data()[i] * sigmoid;
        }
        return result;
    }
}
