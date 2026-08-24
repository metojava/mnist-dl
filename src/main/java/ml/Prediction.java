package ml;

public class Prediction {

    private final int predictedLabel;
    private final int actualLabel;
    private final double confidence;

    public Prediction(
            int predictedLabel,
            int actualLabel,
            double confidence
    ) {
        this.predictedLabel = predictedLabel;
        this.actualLabel = actualLabel;
        this.confidence = confidence;
    }

    public int predictedLabel() {
        return predictedLabel;
    }

    public int actualLabel() {
        return actualLabel;
    }

    public double confidence() {
        return confidence;
    }

    public boolean correct() {
        return predictedLabel == actualLabel;
    }

    public double confidencePercentage() {
        return confidence * 100.0;
    }

    @Override
    public String toString() {
        return String.format(
                "Predicted: %d | Actual: %d | Confidence: %.2f%%",
                predictedLabel,
                actualLabel,
                confidencePercentage()
        );
    }
}
