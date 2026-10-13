package ml;

/**
 * Softmax activation.
 *
 * Computed independently for every example in the batch, subtracting
 * the row maximum for numerical stability.
 *
 * Buffers are reused across calls to avoid per-batch allocations.
 */
public class Softmax implements Activation {

    private Tensor output;
    private Tensor outputBuffer;
    private Tensor gradientBuffer;

    @Override
    public Tensor forward(Tensor input) {

        Tensor out =
                ensure(
                        outputBuffer,
                        input.rows(),
                        input.cols()
                );

        outputBuffer =
                out;

        float[] inputData =
                input.data();

        float[] outputData =
                out.data();

        int rows =
                input.rows();

        int cols =
                input.cols();

        int offset = 0;

        for (int i = 0;
             i < rows;
             i++) {

            float max =
                    Float.NEGATIVE_INFINITY;

            for (int j = 0;
                 j < cols;
                 j++) {

                max = Math.max(
                        max,
                        inputData[offset + j]
                );
            }

            float sum = 0.0f;

            for (int j = 0;
                 j < cols;
                 j++) {

                float value =
                        (float) Math.exp(
                                (double) (inputData[offset + j] - max)
                        );

                outputData[offset + j] =
                        value;

                sum += value;
            }

            for (int j = 0;
                 j < cols;
                 j++) {

                outputData[offset + j] /=
                        sum;
            }

            offset += cols;
        }

        output =
                out;

        return out;
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
                ensure(
                        gradientBuffer,
                        gradient.rows(),
                        gradient.cols()
                );

        gradientBuffer =
                result;

        float[] gradientData =
                gradient.data();

        float[] resultData =
                result.data();

        float[] outputData =
                output.data();

        int cols =
                gradient.cols();

        for (int i = 0;
             i < gradient.rows();
             i++) {

            int base =
                    i * cols;

            for (int j = 0;
                 j < cols;
                 j++) {

                float outputValue =
                        outputData[base + j];

                float value =
                        0.0f;

                for (int k = 0;
                     k < cols;
                     k++) {

                    float delta =
                            j == k ? 1.0f : 0.0f;

                    float derivative =
                            outputValue
                            * (delta - outputData[base + k]);

                    value +=
                            gradientData[base + k]
                            * derivative;
                }

                resultData[base + j] =
                        value;
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

    private static Tensor ensure(
            Tensor tensor,
            int rows,
            int cols
    ) {

        if (tensor == null ||
            tensor.rows() != rows ||
            tensor.cols() != cols) {

            return new Tensor(
                    rows,
                    cols
            );
        }

        return tensor;
    }
}