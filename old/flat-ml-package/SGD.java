package ml;

public class SGD {

    private double learningRate;

    public SGD(double learningRate) {

        if (learningRate <= 0) {
            throw new IllegalArgumentException(
                    "Learning rate must be positive."
            );
        }

        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {

        if (learningRate <= 0) {
            throw new IllegalArgumentException(
                    "Learning rate must be positive."
            );
        }

        this.learningRate = learningRate;
    }
}
