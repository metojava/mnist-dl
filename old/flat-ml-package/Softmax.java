package ml;

public class Softmax implements Activation {

    private Tensor output;

    @Override
    public Tensor forward(Tensor input) {

        output =
                new Tensor(
                        input.rows(),
                        input.cols()
                );

        /*
         * Softmax is calculated independently
         * for every example in the batch.
         *
         * We subtract the maximum value for numerical
         * stability before calculating exp().
         */
        for (int i = 0; i < input.rows(); i++) {

            double max = Double.NEGATIVE_INFINITY;

            for (int j = 0; j < input.cols(); j++) {
                max = Math.max(
                        max,
                        input.get(i, j)
                );
            }

            double sum = 0.0;

            for (int j = 0; j < input.cols(); j++) {

                double value =
                        Math.exp(
                                input.get(i, j) - max
                        );

                output.set(i, j, value);

                sum += value;
            }

            for (int j = 0; j < input.cols(); j++) {

                output.set(
                        i,
                        j,
                        output.get(i, j) / sum
                );
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {

        /*
         * For Softmax + CrossEntropy the gradient is
         * normally calculated directly by CrossEntropyLoss.
         *
         * This implementation provides the general
         * Jacobian-vector product.
         */
        Tensor result =
                new Tensor(
                        gradient.rows(),
                        gradient.cols()
                );

        for (int i = 0; i < output.rows(); i++) {

            for (int j = 0; j < output.cols(); j++) {

                double value = 0.0;

                for (int k = 0; k < output.cols(); k++) {

                    double delta =
                            j == k ? 1.0 : 0.0;

                    double derivative =
                            output.get(i, j)
                            * (delta - output.get(i, k));

                    value +=
                            gradient.get(i, k)
                            * derivative;
                }

                result.set(i, j, value);
            }
        }

        return result;
    }

    @Override
    public void update(double learningRate) {
        // No trainable parameters.
    }

    @Override
    public int inputSize() {
        return -1;
    }

    @Override
    public int outputSize() {
        return -1;
    }
}
