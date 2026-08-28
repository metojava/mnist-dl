package ml.evaluation;

public record Prediction(int predictedLabel, int actualLabel, double confidence) {
    public boolean correct() { return predictedLabel == actualLabel; }
    public double confidencePercentage() { return confidence * 100; }

    @Override
    public String toString() {
        return String.format("Prediction(predicted=%d, actual=%d, confidence=%.2f%%, correct=%s)",
            predictedLabel, actualLabel, confidencePercentage(), correct());
    }
}
