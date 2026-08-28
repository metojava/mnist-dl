package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class GELU extends ActivationLayer {
    private static final double SQRT_2_PI = Math.sqrt(2.0 / Math.PI);

    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> {
            double cdf = 0.5 * (1.0 + Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x)));
            return x * cdf;
        });
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++) {
            double x = lastInput.data()[i];
            double tanhVal = Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x));
            double sech2 = 1.0 - tanhVal * tanhVal;
            double dCdf = 0.5 * SQRT_2_PI * sech2 * (1.0 + 3.0 * 0.044715 * x * x);
            double dGelu = 0.5 * (1.0 + tanhVal) + x * dCdf;
            result.data()[i] = gradient.data()[i] * dGelu;
        }
        return result;
    }
}
