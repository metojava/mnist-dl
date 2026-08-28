package ml.regularization;

import ml.core.Tensor;
import ml.layer.ActivationLayer;
import java.util.Random;

public class Dropout extends ActivationLayer {
    private final double rate;
    private final Random rng;
    private Tensor mask;
    private boolean training;

    public Dropout(double rate, Random rng) {
        this.rate = rate;
        this.rng = rng;
    }

    public Dropout(double rate) {
        this(rate, new Random());
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        this.training = training;
        if (!training) return input;
        mask = Tensor.zeros(input.rows(), input.cols());
        double scale = 1.0 / (1.0 - rate);
        for (int i = 0; i < mask.size(); i++) {
            mask.data()[i] = rng.nextDouble() >= rate ? scale : 0;
        }
        Tensor output = Tensor.zeros(input.rows(), input.cols());
        for (int i = 0; i < input.size(); i++)
            output.data()[i] = input.data()[i] * mask.data()[i];
        lastOutput = output;
        return output;
    }

    @Override
    protected Tensor computeForward(Tensor input) {
        return forward(input, true);
    }

    @Override
    public Tensor backward(Tensor gradient) {
        if (!training || mask == null) return gradient;
        int len = gradient.size();
        Tensor result = new Tensor(gradient.rows(), gradient.cols());
        double[] gd = gradient.data();
        double[] md = mask.data();
        double[] rd = result.data();
        for (int i = 0; i < len; i++)
            rd[i] = gd[i] * md[i];
        return result;
    }

    public double rate() { return rate; }
    public double getRate() { return rate; }
}
