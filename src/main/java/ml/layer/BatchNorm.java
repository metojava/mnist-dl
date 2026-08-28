package ml.layer;

import ml.core.Tensor;
import java.util.Random;

public class BatchNorm implements Layer {
    private final int featureSize;
    private final double momentum;
    private final double epsilon;
    private final Tensor gamma;
    private final Tensor beta;
    private Tensor runningMean;
    private Tensor runningVar;
    private Tensor lastNormalized;
    private Tensor lastInput;
    private final Random rng;

    public BatchNorm(int featureSize, double momentum, double epsilon, Random rng) {
        this.featureSize = featureSize;
        this.momentum = momentum;
        this.epsilon = epsilon;
        this.rng = rng;
        this.gamma = Tensor.ones(1, featureSize);
        this.beta = Tensor.zeros(1, featureSize);
        this.runningMean = Tensor.zeros(1, featureSize);
        this.runningVar = Tensor.ones(1, featureSize);
    }

    public BatchNorm(int featureSize) {
        this(featureSize, 0.1, 1e-5, new Random());
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        lastInput = input;
        if (training) {
            Tensor mean = input.sumCols().multiply(1.0 / input.rows());
            Tensor diff = input.subtract(mean);
            Tensor sqDiff = diff.multiply(diff);
            Tensor variance = sqDiff.sumCols().multiply(1.0 / input.rows());
            for (int i = 0; i < featureSize; i++) {
                runningMean.data()[i] = momentum * runningMean.data()[i] + (1 - momentum) * mean.data()[i];
                runningVar.data()[i] = momentum * runningVar.data()[i] + (1 - momentum) * variance.data()[i];
            }
            Tensor norm = Tensor.zeros(input.rows(), featureSize);
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < featureSize; j++) {
                    double std = Math.sqrt(runningVar.data()[j] + epsilon);
                    norm.data()[i * featureSize + j] = (input.data()[i * featureSize + j] - runningMean.data()[j]) / std;
                }
            }
            lastNormalized = norm;
        } else {
            Tensor norm = Tensor.zeros(input.rows(), featureSize);
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < featureSize; j++) {
                    double std = Math.sqrt(runningVar.data()[j] + epsilon);
                    norm.data()[i * featureSize + j] = (input.data()[i * featureSize + j] - runningMean.data()[j]) / std;
                }
            }
            lastNormalized = norm;
        }
        Tensor output = Tensor.zeros(input.rows(), featureSize);
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < featureSize; j++) {
                output.data()[i * featureSize + j] = gamma.data()[j] * lastNormalized.data()[i * featureSize + j] + beta.data()[j];
            }
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        return gradient;
    }

    @Override
    public void update(double learningRate) {
        for (int j = 0; j < featureSize; j++) {
            gamma.data()[j] -= learningRate * 0.001;
            beta.data()[j] -= learningRate * 0.001;
        }
    }

    @Override
    public int inputSize() { return featureSize; }
    @Override
    public int outputSize() { return featureSize; }
    @Override
    public int parameterCount() { return featureSize * 2; }

    public Tensor gamma() { return gamma; }
    public Tensor beta() { return beta; }
    public Tensor runningMean() { return runningMean; }
    public Tensor runningVar() { return runningVar; }
}
