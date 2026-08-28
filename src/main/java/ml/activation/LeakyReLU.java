package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class LeakyReLU extends ActivationLayer {
    private final double alpha;

    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }

    public LeakyReLU() {
        this(0.01);
    }

    public double alpha() { return alpha; }

    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> x > 0 ? x : alpha * x);
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++)
            result.data()[i] = lastInput.data()[i] > 0 ? gradient.data()[i] : alpha * gradient.data()[i];
        return result;
    }
}
