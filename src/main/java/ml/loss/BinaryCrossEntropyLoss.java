package ml.loss;

import ml.core.Tensor;

public class BinaryCrossEntropyLoss implements Loss {
    private final double epsilon;

    public BinaryCrossEntropyLoss(double epsilon) {
        this.epsilon = epsilon;
    }

    public BinaryCrossEntropyLoss() {
        this(1e-7);
    }

    @Override
    public double forward(Tensor predictions, int[] labels) {
        double loss = 0;
        int batchSize = labels.length;
        int classes = predictions.cols();
        for (int i = 0; i < batchSize; i++) {
            int label = labels[i];
            double p = Math.max(Math.min(predictions.get(i, label), 1 - epsilon), epsilon);
            loss -= Math.log(p);
        }
        return loss / batchSize;
    }

    public double forward(Tensor predictions, Tensor targets) {
        double loss = 0;
        for (int i = 0; i < predictions.size(); i++) {
            double p = Math.max(Math.min(predictions.data()[i], 1 - epsilon), epsilon);
            double t = targets.data()[i];
            loss -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
        }
        return loss / predictions.rows();
    }

    @Override
    public Tensor backward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int classes = predictions.cols();
        Tensor grad = Tensor.zeros(batchSize, classes);
        for (int i = 0; i < batchSize; i++) {
            int label = labels[i];
            for (int j = 0; j < classes; j++) {
                double target = (j == label) ? 1.0 : 0.0;
                double p = Math.max(Math.min(predictions.get(i, j), 1 - epsilon), epsilon);
                grad.data()[i * classes + j] = (-target / p + (1 - target) / (1 - p)) / batchSize;
            }
        }
        return grad;
    }

    public Tensor backward(Tensor predictions, Tensor targets) {
        int rows = predictions.rows();
        int cols = predictions.cols();
        Tensor grad = Tensor.zeros(rows, cols);
        for (int i = 0; i < predictions.size(); i++) {
            double p = Math.max(Math.min(predictions.data()[i], 1 - epsilon), epsilon);
            double t = targets.data()[i];
            grad.data()[i] = (-t / p + (1 - t) / (1 - p)) / rows;
        }
        return grad;
    }
}
