package ml;

import java.util.Random;

/**
 * Fully connected (dense) neural network layer.
 *
 * Forward:
 *
 *     Z = XW + b
 *
 * Backward:
 *
 *     dX = dZ W^T
 *     dW = X^T dZ
 *     db = sum(dZ)
 *
 * The layer stores the input from the forward pass so that
 * it can be used during backpropagation.
 *
 * All intermediate tensors are kept in reusable buffers that are
 * only reallocated when the batch size changes, so no per-batch
 * garbage is produced.
 */
public class DenseLayer implements Layer {

    private final int inputSize;
    private final int outputSize;

    /*
     * Weights:
     *
     * inputSize x outputSize
     */
    private final Tensor weights;

    /*
     * Bias:
     *
     * 1 x outputSize
     */
    private final Tensor bias;

    /*
     * Gradients accumulated during backward().
     */
    private Tensor weightsGradient;
    private Tensor biasGradient;

    /*
     * Input saved during forward().
     */
    private Tensor lastInput;

    /*
     * Cached transpose of the weights:
     *
     * outputSize x inputSize
     *
     * Rebuilt lazily and invalidated whenever update() changes
     * the weights, so it is computed once per batch instead of
     * materializing W^T on every backward call.
     */
    private Tensor weightsTransposed;

    /*
     * Reusable output / gradient buffers.
     */
    private Tensor outputBuffer;
    private Tensor inputGradientBuffer;

    private static final Random RANDOM =
            new Random();

    // ============================================================
    // Constructor
    // ============================================================

    public DenseLayer(
            int inputSize,
            int outputSize
    ) {

        if (inputSize <= 0) {
            throw new IllegalArgumentException(
                    "inputSize must be greater than zero."
            );
        }

        if (outputSize <= 0) {
            throw new IllegalArgumentException(
                    "outputSize must be greater than zero."
            );
        }

        this.inputSize =
                inputSize;

        this.outputSize =
                outputSize;

        /*
         * He initialization:
         *
         * std = sqrt(2 / inputSize)
         *
         * This works particularly well with ReLU.
         */
        double std =
                Math.sqrt(
                        2.0 / inputSize
                );

        this.weights =
                new Tensor(
                        inputSize,
                        outputSize
                );

        this.bias =
                new Tensor(
                        1,
                        outputSize
                );

        float[] weightData =
                weights.data();

        for (int i = 0;
             i < weightData.length;
             i++) {

            weightData[i] =
                    (float) (RANDOM.nextGaussian()
                            * std);
        }

        /*
         * Bias starts at zero.
         */
        bias.fill(0.0f);
    }

    // ============================================================
    // Forward
    // ============================================================

    /**
     * Forward pass:
     *
     *     output = input × weights + bias
     *
     * Input:
     *
     *     batchSize × inputSize
     *
     * Weights:
     *
     *     inputSize × outputSize
     *
     * Output:
     *
     *     batchSize × outputSize
     */
    @Override
    public Tensor forward(
            Tensor input
    ) {

        if (input.cols() != inputSize) {
            throw new IllegalArgumentException(
                    "Expected input with " +
                    inputSize +
                    " columns, but received " +
                    input.cols()
            );
        }

        /*
         * Save input for backpropagation.
         */
        lastInput =
                input;

        /*
         * Matrix multiplication into a reused buffer.
         */
        Tensor output =
                ensure(
                        outputBuffer,
                        input.rows(),
                        outputSize
                );

        outputBuffer =
                output;

        Tensor.matmul(
                input,
                weights,
                output
        );

        /*
         * Add bias to every row.
         *
         * Doing this directly on the flat array is much faster
         * than repeatedly creating temporary tensors.
         */
        float[] outputData =
                output.data();

        float[] biasData =
                bias.data();

        int batchSize =
                input.rows();

        for (int i = 0;
             i < batchSize;
             i++) {

            int outputOffset =
                    i * outputSize;

            for (int j = 0;
                 j < outputSize;
                 j++) {

                outputData[
                        outputOffset + j
                ] += biasData[j];
            }
        }

        return output;
    }

    // ============================================================
    // Backward
    // ============================================================

    /**
     * Backpropagation.
     *
     * Given:
     *
     *     dZ = gradient from the next layer
     *
     * Calculates:
     *
     *     dW = X^T × dZ
     *     db = sum(dZ)
     *     dX = dZ × W^T
     */
    @Override
    public Tensor backward(
            Tensor gradient
    ) {

        if (lastInput == null) {
            throw new IllegalStateException(
                    "Cannot call backward() before forward()."
            );
        }

        if (gradient.cols() != outputSize) {
            throw new IllegalArgumentException(
                    "Gradient must have " +
                    outputSize +
                    " columns, but received " +
                    gradient.cols()
            );
        }

        if (gradient.rows() != lastInput.rows()) {
            throw new IllegalArgumentException(
                    "Gradient batch size does not match input batch size."
            );
        }

        /*
         * --------------------------------------------------------
         * Weight gradient
         * --------------------------------------------------------
         *
         * dW = X^T × dZ
         *
         * X:
         *     batchSize × inputSize
         *
         * X^T:
         *     inputSize × batchSize
         *
         * dZ:
         *     batchSize × outputSize
         *
         * dW:
         *     inputSize × outputSize
         *
         * The transposed variant reads X with A^T semantics
         * directly, without materializing a copy of X^T.
         */
        weightsGradient =
                ensure(
                        weightsGradient,
                        inputSize,
                        outputSize
                );

        Tensor.matmulTransposedA(
                lastInput,
                gradient,
                weightsGradient
        );

        /*
         * --------------------------------------------------------
         * Bias gradient
         * --------------------------------------------------------
         *
         * db = sum(dZ) over batch dimension
         */
        biasGradient =
                ensure(
                        biasGradient,
                        1,
                        outputSize
                );

        float[] gradientData =
                gradient.data();

        float[] biasGradientData =
                biasGradient.data();

        java.util.Arrays.fill(
                biasGradientData,
                0.0f
        );

        int batchSize =
                gradient.rows();

        for (int i = 0;
             i < batchSize;
             i++) {

            int offset =
                    i * outputSize;

            for (int j = 0;
                 j < outputSize;
                 j++) {

                biasGradientData[j] +=
                        gradientData[
                                offset + j
                        ];
            }
        }

        /*
         * --------------------------------------------------------
         * Input gradient
         * --------------------------------------------------------
         *
         * dX = dZ × W^T
         *
         * dZ:
         *     batchSize × outputSize
         *
         * W^T:
         *     outputSize × inputSize
         *
         * dX:
         *     batchSize × inputSize
         */
        inputGradientBuffer =
                ensure(
                        inputGradientBuffer,
                        gradient.rows(),
                        inputSize
                );

        Tensor.matmul(
                gradient,
                weightsTransposed(),
                inputGradientBuffer
        );

        return inputGradientBuffer;
    }

    // ============================================================
    // Parameter update
    // ============================================================

    /**
     * Updates weights and bias using the supplied learning rate.
     */
    @Override
    public void update(
            double learningRate
    ) {

        if (weightsGradient == null ||
            biasGradient == null) {

            throw new IllegalStateException(
                    "Cannot update before backward()."
            );
        }

        float[] weightData =
                weights.data();

        float[] weightGradientData =
                weightsGradient.data();

        float[] biasData =
                bias.data();

        float[] biasGradientData =
                biasGradient.data();

        float learningRateFloat =
                (float) learningRate;

        /*
         * Weight update:
         *
         *     W = W - learningRate * dW
         */
        for (int i = 0;
             i < weightData.length;
             i++) {

            weightData[i] -=
                    learningRateFloat *
                    weightGradientData[i];
        }

        /*
         * Bias update:
         *
         *     b = b - learningRate * db
         */
        for (int i = 0;
             i < biasData.length;
             i++) {

            biasData[i] -=
                    learningRateFloat *
                    biasGradientData[i];
        }

        /*
         * Weights changed, so the cached transpose is stale.
         */
        weightsTransposed =
                null;
    }

    // ============================================================
    // Helpers
    // ============================================================

    /*
     * Returns W^T, rebuilding it only when the weights have
     * changed (i.e. since the last update()).
     */
    private Tensor weightsTransposed() {

        if (weightsTransposed == null) {

            weightsTransposed =
                    weights.transpose();
        }

        return weightsTransposed;
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

    // ============================================================
    // Getters
    // ============================================================

    public Tensor weights() {
        return weights;
    }

    public Tensor bias() {
        return bias;
    }

    public Tensor weightsGradient() {
        return weightsGradient;
    }

    public Tensor biasGradient() {
        return biasGradient;
    }

    public int inputSize() {
        return inputSize;
    }

    public int outputSize() {
        return outputSize;
    }

    // ============================================================
    // Parameter count
    // ============================================================

    public int parameterCount() {

        return
                inputSize * outputSize
                + outputSize;
    }

    // ============================================================
    // String representation
    // ============================================================

    @Override
    public String toString() {

        return String.format(
                "Dense(%d -> %d, parameters=%d)",
                inputSize,
                outputSize,
                parameterCount()
        );
    }
}