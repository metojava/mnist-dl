package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class ELU extends ActivationLayer {
    private final double alpha;

    public ELU(double alpha) {
        this.alpha = alpha;
    }

    public ELU() {
        this(1.0);
    }

    public double alpha() { return alpha; }

    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> x > 0 ? x : alpha * (Math.exp(x) - 1));
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++) {
            double x = lastInput.data()[i];
            result.data()[i] = x > 0 ? gradient.data()[i] : gradient.data()[i] * alpha * Math.exp(x);
        }
        return result;
    }
}
