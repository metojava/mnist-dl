package ml.layer;

import ml.core.Tensor;

public class LayerNorm implements Layer {
    private final int inputSize;
    private final Tensor gamma;
    private final Tensor beta;
    private final double epsilon;

    private final double[] gradGamma;
    private final double[] gradBeta;

    private Tensor lastInput;
    private Tensor lastXhat;
    private double[] lastMean;
    private double[] lastVar;

    public LayerNorm(int inputSize, double epsilon) {
        this.inputSize = inputSize;
        this.epsilon = epsilon;
        this.gamma = Tensor.ones(1, inputSize);
        this.beta = Tensor.zeros(1, inputSize);
        this.gradGamma = new double[inputSize];
        this.gradBeta = new double[inputSize];
    }

    public LayerNorm(int inputSize) {
        this(inputSize, 1e-5);
    }

    public Tensor gamma() { return gamma; }
    public Tensor beta() { return beta; }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        lastInput = input;
        int batchSize = input.rows();
        int features = input.cols();

        Tensor output = new Tensor(batchSize, features);
        lastXhat = new Tensor(batchSize, features);
        lastMean = new double[batchSize];
        lastVar = new double[batchSize];

        for (int b = 0; b < batchSize; b++) {
            double mean = 0;
            for (int f = 0; f < features; f++) mean += input.get(b, f);
            mean /= features;
            double var = 0;
            for (int f = 0; f < features; f++) {
                double d = input.get(b, f) - mean;
                var += d * d;
            }
            var /= features;

            lastMean[b] = mean;
            lastVar[b] = var;

            double invStd = 1.0 / Math.sqrt(var + epsilon);
            for (int f = 0; f < features; f++) {
                double xhat = (input.get(b, f) - mean) * invStd;
                lastXhat.set(b, f, xhat);
                output.set(b, f, gamma.data()[f] * xhat + beta.data()[f]);
            }
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int batchSize = gradient.rows();
        int features = gradient.cols();
        Tensor dInput = new Tensor(batchSize, features);

        for (int b = 0; b < batchSize; b++) {
            double invStd = 1.0 / Math.sqrt(lastVar[b] + epsilon);
            double sumDxhat = 0;
            double sumDxhatXhat = 0;

            for (int f = 0; f < features; f++) {
                double dxhat = gradient.get(b, f) * gamma.data()[f];
                sumDxhat += dxhat;
                sumDxhatXhat += dxhat * lastXhat.get(b, f);
                gradGamma[f] += gradient.get(b, f) * lastXhat.get(b, f);
                gradBeta[f] += gradient.get(b, f);
            }

            double normSumDxhat = sumDxhat / features;
            double normSumDxhatXhat = sumDxhatXhat / features;

            for (int f = 0; f < features; f++) {
                double dxhat = gradient.get(b, f) * gamma.data()[f];
                double dx = invStd * (dxhat - normSumDxhat - lastXhat.get(b, f) * normSumDxhatXhat);
                dInput.set(b, f, dx);
            }
        }
        return dInput;
    }

    @Override
    public void update(double learningRate) {
        for (int i = 0; i < inputSize; i++) {
            gamma.data()[i] -= learningRate * gradGamma[i];
            beta.data()[i] -= learningRate * gradBeta[i];
        }
        java.util.Arrays.fill(gradGamma, 0.0);
        java.util.Arrays.fill(gradBeta, 0.0);
    }

    @Override
    public int inputSize() { return inputSize; }

    @Override
    public int outputSize() { return inputSize; }

    @Override
    public int parameterCount() { return 2 * inputSize; }
}
