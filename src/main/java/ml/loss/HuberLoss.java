package ml.loss;

import ml.core.Tensor;

public class HuberLoss implements Loss {
    private final double delta;

    public HuberLoss(double delta) {
        this.delta = delta;
    }

    public HuberLoss() {
        this(1.0);
    }

    @Override
    public double forward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int classes = predictions.cols();
        double loss = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < classes; j++) {
                double target = (j == labels[i]) ? 1.0 : 0.0;
                double diff = Math.abs(predictions.get(i, j) - target);
                if (diff <= delta) {
                    loss += 0.5 * diff * diff;
                } else {
                    loss += delta * diff - 0.5 * delta * delta;
                }
            }
        }
        return loss / batchSize;
    }

    @Override
    public Tensor backward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int classes = predictions.cols();
        Tensor grad = Tensor.zeros(batchSize, classes);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < classes; j++) {
                double target = (j == labels[i]) ? 1.0 : 0.0;
                double diff = predictions.get(i, j) - target;
                if (Math.abs(diff) <= delta) {
                    grad.data()[i * classes + j] = (float) (diff / batchSize);
                } else {
                    grad.data()[i * classes + j] = (float) (delta * Math.signum(diff) / batchSize);
                }
            }
        }
        return grad;
    }
}
