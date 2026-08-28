package ml;

public class EvaluationResult {

    private final int total;
    private final int correct;
    private final ConfusionMatrix confusionMatrix;

    public EvaluationResult(
            int total,
            int correct,
            ConfusionMatrix confusionMatrix
    ) {
        this.total = total;
        this.correct = correct;
        this.confusionMatrix = confusionMatrix;
    }

    public int total() {
        return total;
    }

    public int correct() {
        return correct;
    }

    public int incorrect() {
        return total - correct;
    }

    public double accuracy() {

        if (total == 0) {
            return 0.0;
        }

        return (double) correct / total;
    }

    public double accuracyPercentage() {
        return accuracy() * 100.0;
    }

    public ConfusionMatrix confusionMatrix() {
        return confusionMatrix;
    }

    @Override
    public String toString() {

        return String.format(
                "Total: %d%n" +
                "Correct: %d%n" +
                "Incorrect: %d%n" +
                "Accuracy: %.2f%%",
                total,
                correct,
                incorrect(),
                accuracyPercentage()
        );
    }
}
