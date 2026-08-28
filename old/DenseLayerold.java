package ml;

import java.util.Random;

public class DenseLayer implements Layer {

    private final int inputSize;
    private final int outputSize;

    private final Tensor weights;
    private final Tensor biases;

    private Tensor input;

    private Tensor weightGradients;
    private Tensor biasGradients;

    public DenseLayer(int inputSize, int outputSize) {
        if (inputSize <= 0 || outputSize <= 0) {
            throw new IllegalArgumentException(
                    "Layer dimensions must be positive."
            );
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;

        /*
         * He/Xavier-style initialization.
         *
         * This is useful for neural networks because
         * starting every weight at zero would prevent
         * neurons from learning different things.
         */
        double stddev = Math.sqrt(2.0 / inputSize);

        Random random = new Random();

        weights = new Tensor(inputSize, outputSize);

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights.set(
                        i,
                        j,
                        random.nextGaussian() * stddev
                );
            }
        }

        biases = Tensor.zeros(1, outputSize);
    }

    @Override
    public Tensor forward(Tensor input) {

        if (input.cols() != inputSize) {
            throw new IllegalArgumentException(
                    "Expected input with " +
                    inputSize +
                    " features, got " +
                    input.cols()
            );
        }

        this.input = input;

        return input.matmul(weights).add(
                broadcastBias(input.rows())
        );
    }

    @Override
    public Tensor backward(Tensor gradient) {

        /*
         * dW = X^T * dY
         */
        weightGradients =
                input.transpose().matmul(gradient);

        /*
         * db = sum(dY over batch)
         */
        biasGradients =
                gradient.sumRows();

        /*
         * dX = dY * W^T
         */
        return gradient.matmul(weights.transpose());
    }

    @Override
    public void update(double learningRate) {

        for (int i = 0; i < weights.rows(); i++) {
            for (int j = 0; j < weights.cols(); j++) {

                double newValue =
                        weights.get(i, j)
                        - learningRate * weightGradients.get(i, j);

                weights.set(i, j, newValue);
            }
        }

        for (int j = 0; j < biases.cols(); j++) {

            double newValue =
                    biases.get(0, j)
                    - learningRate * biasGradients.get(0, j);

            biases.set(0, j, newValue);
        }
    }

    private Tensor broadcastBias(int rows) {

        Tensor result = new Tensor(rows, outputSize);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < outputSize; j++) {
                result.set(
                        i,
                        j,
                        biases.get(0, j)
                );
            }
        }

        return result;
    }

    @Override
    public int inputSize() {
        return inputSize;
    }

    @Override
    public int outputSize() {
        return outputSize;
    }
}
