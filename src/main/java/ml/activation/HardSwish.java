package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class HardSwish extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> {
            double relu6 = Math.min(Math.max(x + 3.0, 0.0), 6.0);
            return x * relu6 / 6.0;
        });
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++) {
            double x = lastInput.data()[i];
            double dHardSwish;
            if (x <= -3.0) {
                dHardSwish = 0.0;
            } else if (x < 3.0) {
                dHardSwish = (2.0 * x + 3.0) / 6.0;
            } else {
                dHardSwish = 1.0;
            }
            result.data()[i] = gradient.data()[i] * dHardSwish;
        }
        return result;
    }
}
