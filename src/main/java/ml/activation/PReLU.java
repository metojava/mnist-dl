package ml.activation;

import ml.core.Tensor;
import ml.exception.ShapeException;
import ml.layer.ActivationLayer;

import java.util.Random;

public class PReLU extends ActivationLayer {
    private final double alphaInit;
    private double alpha;
    private double alphaGrad;

    public PReLU(double alphaInit) {
        this.alphaInit = alphaInit;
        this.alpha = alphaInit;
    }

    public PReLU() {
        this(0.25);
    }

    public double alpha() { return alpha; }

    public void setAlpha(double alpha) { this.alpha = alpha; }

    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> x > 0 ? x : alpha * x);
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int len = lastInput.size();
        Tensor result = new Tensor(lastInput.rows(), lastInput.cols());
        double gradSum = 0;
        for (int i = 0; i < len; i++) {
            double x = lastInput.data()[i];
            if (x > 0) {
                result.data()[i] = gradient.data()[i];
            } else {
                result.data()[i] = alpha * gradient.data()[i];
                gradSum += gradient.data()[i] * x;
            }
        }
        this.alphaGrad = gradSum / len;
        return result;
    }

    @Override
    public void update(double learningRate) {
        alpha -= learningRate * alphaGrad;
    }

    @Override
    public int parameterCount() { return 1; }
}
