package ml.evaluation;

public record PredictionResult(Prediction[] predictions) {
    public int size() { return predictions.length; }
    public Prediction get(int index) { return predictions[index]; }

    public int correct() {
        int count = 0;
        for (Prediction p : predictions) if (p.correct()) count++;
        return count;
    }

    public int incorrect() { return size() - correct(); }
    public double accuracy() { return (double) correct() / size(); }
    public double accuracyPercentage() { return accuracy() * 100; }

    @Override
    public String toString() {
        return String.format("PredictionResult(accuracy=%.2f%%, correct=%d/%d)",
            accuracyPercentage(), correct(), size());
    }
}
