package ml;

public class CrossEntropyLoss implements Loss {

    private static final double EPSILON = 1e-12;

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

        double totalLoss = 0.0;

        for (int i = 0; i < predictions.rows(); i++) {

            int label = labels[i];

            if (label < 0 || label >= predictions.cols()) {
                throw new IllegalArgumentException(
                        "Invalid label: " + label
                );
            }

            double probability =
                    Math.max(
                            predictions.get(i, label),
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

        Tensor gradient =
                new Tensor(
                        predictions.rows(),
                        predictions.cols()
                );

        double batchSize = predictions.rows();

        /*
         * Gradient of softmax + cross entropy:
         *
         * prediction - oneHot(label)
         *
         * divided by batch size.
         */
        for (int i = 0; i < predictions.rows(); i++) {

            for (int j = 0; j < predictions.cols(); j++) {

                double value =
                        predictions.get(i, j);

                if (j == labels[i]) {
                    value -= 1.0;
                }

                gradient.set(
                        i,
                        j,
                        value / batchSize
                );
            }
        }

        return gradient;
    }
}
