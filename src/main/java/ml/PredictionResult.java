package ml;

public class PredictionResult {

    private final Prediction[] predictions;

    public PredictionResult(Prediction[] predictions) {
        this.predictions = predictions;
    }

    public Prediction[] predictions() {
        return predictions;
    }

    public Prediction get(int index) {
        return predictions[index];
    }

    public int size() {
        return predictions.length;
    }

    public int correct() {

        int count = 0;

        for (Prediction prediction : predictions) {
            if (prediction.correct()) {
                count++;
            }
        }

        return count;
    }

    public double accuracy() {

        if (predictions.length == 0) {
            return 0.0;
        }

        return (double) correct() / predictions.length;
    }

    public double accuracyPercentage() {
        return accuracy() * 100.0;
    }
}
