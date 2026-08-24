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

        double[] weightData =
                weights.data();

        for (int i = 0;
             i < weightData.length;
             i++) {

            weightData[i] =
                    RANDOM.nextGaussian()
                            * std;
        }

        /*
         * Bias starts at zero.
         */
        bias.fill(0.0);
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
         * Matrix multiplication.
         *
         * This uses the optimized flat-array implementation
         * in Tensor.matmul().
         */
        Tensor output =
                Tensor.matmul(
                        input,
                        weights
                );

        /*
         * Add bias to every row.
         *
         * Doing this directly on the flat array is much faster
         * than repeatedly creating temporary tensors.
         */
        double[] outputData =
                output.data();

        double[] biasData =
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
         */
        weightsGradient =
                Tensor.matmul(
                        lastInput.transpose(),
                        gradient
                );

        /*
         * --------------------------------------------------------
         * Bias gradient
         * --------------------------------------------------------
         *
         * db = sum(dZ) over batch dimension
         */
        biasGradient =
                new Tensor(
                        1,
                        outputSize
                );

        double[] gradientData =
                gradient.data();

        double[] biasGradientData =
                biasGradient.data();

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
        Tensor inputGradient =
                Tensor.matmul(
                        gradient,
                        weights.transpose()
                );

        return inputGradient;
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

        double[] weightData =
                weights.data();

        double[] weightGradientData =
                weightsGradient.data();

        double[] biasData =
                bias.data();

        double[] biasGradientData =
                biasGradient.data();

        /*
         * Weight update:
         *
         *     W = W - learningRate * dW
         */
        for (int i = 0;
             i < weightData.length;
             i++) {

            weightData[i] -=
                    learningRate *
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
                    learningRate *
                    biasGradientData[i];
        }
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
