package ml;

/**
 * Categorical cross-entropy loss:
 *
 *     L = -(1 / N) * sum(log(p[label]))
 *
 * The backward pass returns the fused Softmax + CrossEntropy
 * gradient:
 *
 *     dL/dz = p - oneHot(label)
 *
 * divided by the batch size. The gradient buffer is reused across
 * calls to avoid per-batch allocations.
 */
public class CrossEntropyLoss implements Loss {

    private static final double EPSILON = 1e-12;

    private Tensor gradientBuffer;

    @Override
    public double forward(
            Tensor predictions,
            int[] labels
    ) {

        if (predictions.rows() != labels.length) {
            throw new IllegalArgumentException(
                    "Number of predictions must equal number of labels."
            );
        }

        float[] predictionData =
                predictions.data();

        int cols =
                predictions.cols();

        double totalLoss = 0.0;

        for (int i = 0;
             i < predictions.rows();
             i++) {

            int label = labels[i];

            if (label < 0 || label >= cols) {
                throw new IllegalArgumentException(
                        "Invalid label: " + label
                );
            }

            double probability =
                    Math.max(
                            (double) predictionData[i * cols + label],
                            EPSILON
                    );

            totalLoss -= Math.log(probability);
        }

        return totalLoss / predictions.rows();
    }

    @Override
    public Tensor backward(
            Tensor predictions,
            int[] labels
    ) {

        int rows =
                predictions.rows();

        int cols =
                predictions.cols();

        Tensor gradient =
                ensure(
                        gradientBuffer,
                        rows,
                        cols
                );

        gradientBuffer =
                gradient;

        float[] gradientData =
                gradient.data();

        float[] predictionData =
                predictions.data();

        float scale =
                1.0f / rows;

        for (int i = 0;
             i < rows;
             i++) {

            int label =
                    labels[i];

            int offset =
                    i * cols;

            for (int j = 0;
                 j < cols;
                 j++) {

                float value =
                        predictionData[offset + j];

                if (j == label) {
                    value -= 1.0f;
                }

                gradientData[offset + j] =
                        value * scale;
            }
        }

        return gradient;
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