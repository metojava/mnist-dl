package ml;

public class Predictor {

    private final NeuralNetwork network;

    public Predictor(NeuralNetwork network) {
        this.network = network;
    }

    public Prediction predict(
            Tensor image,
            int actualLabel
    ) {

        Tensor probabilities =
                network.forward(image);

        int predicted =
                probabilities.argmaxRow(0);

        double confidence =
                probabilities.get(0, predicted);

        return new Prediction(
                predicted,
                actualLabel,
                confidence
        );
    }

    public PredictionResult predict(
            Tensor images,
            int[] labels
    ) {

        if (images.rows() != labels.length) {
            throw new IllegalArgumentException(
                    "Number of images must equal number of labels."
            );
        }

        Tensor probabilities =
                network.forward(images);

        Prediction[] predictions =
                new Prediction[images.rows()];

        for (int i = 0; i < images.rows(); i++) {

            int predicted =
                    probabilities.argmaxRow(i);

            double confidence =
                    probabilities.get(i, predicted);

            predictions[i] =
                    new Prediction(
                            predicted,
                            labels[i],
                            confidence
                    );
        }

        return new PredictionResult(
                predictions
        );
    }
}
