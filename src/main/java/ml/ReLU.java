package ml;

/**
 * ReLU activation.
 *
 * Forward:
 *
 *     y = max(0, x)
 *
 * Backward:
 *
 *     dL/dx = dL/dy   if x > 0
 *             0       otherwise
 *
 * The output buffer is reused across calls to avoid per-batch
 * allocations.
 */
public class ReLU implements Activation {

    private Tensor input;
    private Tensor outputBuffer;
    private Tensor gradientBuffer;

    @Override
    public Tensor forward(Tensor input) {

        this.input = input;

        Tensor output =
                ensure(
                        outputBuffer,
                        input.rows(),
                        input.cols()
                );

        outputBuffer =
                output;

        float[] inputData =
                input.data();

        float[] outputData =
                output.data();

        for (int i = 0;
             i < outputData.length;
             i++) {

            float value =
                    inputData[i];

            outputData[i] =
                    value > 0.0f
                            ? value
                            : 0.0f;
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {

        Tensor result =
                ensure(
                        gradientBuffer,
                        gradient.rows(),
                        gradient.cols()
                );

        gradientBuffer =
                result;

        float[] inputData =
                input.data();

        float[] gradientData =
                gradient.data();

        float[] resultData =
                result.data();

        for (int i = 0;
             i < resultData.length;
             i++) {

            resultData[i] =
                    inputData[i] > 0.0f
                            ? gradientData[i]
                            : 0.0f;
        }

        return result;
    }

    @Override
    public void update(double learningRate) {
        // ReLU has no trainable parameters.
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