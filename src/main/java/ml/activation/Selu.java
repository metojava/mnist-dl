package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Selu extends ActivationLayer {
    private static final double ALPHA = 1.6732632423543772848170429916717;
    private static final double SCALE = 1.0507009873554804934193349852946;

    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> x > 0 ? SCALE * x : SCALE * ALPHA * (Math.exp(x) - 1));
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++) {
            double x = lastInput.data()[i];
            result.data()[i] = x > 0
                ? gradient.data()[i] * SCALE
                : gradient.data()[i] * SCALE * ALPHA * Math.exp(x);
        }
        return result;
    }
}
