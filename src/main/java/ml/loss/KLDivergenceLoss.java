package ml.loss;

import ml.core.Tensor;

public class KLDivergenceLoss implements Loss {
    private final double epsilon;

    public KLDivergenceLoss(double epsilon) {
        this.epsilon = epsilon;
    }

    public KLDivergenceLoss() {
        this(1e-12);
    }

    @Override
    public double forward(Tensor predictions, int[] labels) {
        double loss = 0;
        int batchSize = labels.length;
        int classes = predictions.cols();
        for (int i = 0; i < batchSize; i++) {
            int label = labels[i];
            double p = Math.max(predictions.get(i, label), epsilon);
            loss -= Math.log(p);
        }
        return loss / batchSize;
    }

    public double forward(Tensor predictions, Tensor targets) {
        double loss = 0;
        int rows = predictions.rows();
        int cols = predictions.cols();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double q = Math.max(targets.get(i, j), epsilon);
                double p = Math.max(predictions.get(i, j), epsilon);
                loss += q * (Math.log(q) - Math.log(p));
            }
        }
        return loss / rows;
    }

    @Override
    public Tensor backward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int classes = predictions.cols();
        double scale = 1.0 / batchSize;
        Tensor grad = new Tensor(batchSize, classes);
        for (int i = 0; i < batchSize; i++) {
            int label = labels[i];
            for (int j = 0; j < classes; j++) {
                double p = Math.max(predictions.get(i, j), epsilon);
                if (j == label) {
                    grad.data()[i * classes + j] = (float) (-1.0 / p * scale);
                }
            }
        }
        return grad;
    }

    public Tensor backward(Tensor predictions, Tensor targets) {
        int rows = predictions.rows();
        int cols = predictions.cols();
        Tensor grad = Tensor.zeros(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double q = Math.max(targets.get(i, j), epsilon);
                double p = Math.max(predictions.get(i, j), epsilon);
                grad.data()[i * cols + j] = (float) (-q / p / rows);
            }
        }
        return grad;
    }
}
