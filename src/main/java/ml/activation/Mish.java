package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Mish extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        return input.apply(x -> x * Math.tanh(Math.log(1.0 + Math.exp(x))));
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor result = Tensor.zeros(lastInput.rows(), lastInput.cols());
        for (int i = 0; i < lastInput.size(); i++) {
            double x = lastInput.data()[i];
            double sp = Math.log(1.0 + Math.exp(x));
            double tanhSp = Math.tanh(sp);
            double sigm = 1.0 / (1.0 + Math.exp(-x));
            double dSp = sigm;
            double dtanh = 1.0 - tanhSp * tanhSp;
            double dMish = tanhSp + x * sigm * dtanh;
            result.data()[i] = gradient.data()[i] * dMish;
        }
        return result;
    }
}
