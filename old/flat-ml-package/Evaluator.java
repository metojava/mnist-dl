package ml;

public class Evaluator {

    private Evaluator() {
    }

    public static EvaluationResult evaluate(
            NeuralNetwork network,
            Tensor images,
            int[] labels
    ) {

        if (images.rows() != labels.length) {
            throw new IllegalArgumentException(
                    "Number of images must equal number of labels."
            );
        }

        Tensor predictions =
                network.forward(images);

        ConfusionMatrix confusionMatrix =
                new ConfusionMatrix(10);

        int correct = 0;

        for (int i = 0;
             i < images.rows();
             i++) {

            int predicted =
                    predictions.argmaxRow(i);

            int actual =
                    labels[i];

            confusionMatrix.add(
                    actual,
                    predicted
            );

            if (predicted == actual) {
                correct++;
            }
        }

        return new EvaluationResult(
                images.rows(),
                correct,
                confusionMatrix
        );
    }
}
