package ml.loss;

import ml.core.Tensor;

public class MSELoss implements Loss {
    @Override
    public double forward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int classes = predictions.cols();
        double loss = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < classes; j++) {
                double target = (j == labels[i]) ? 1.0 : 0.0;
                double diff = predictions.get(i, j) - target;
                loss += diff * diff;
            }
        }
        return loss / batchSize;
    }

    public double forward(Tensor predictions, Tensor targets) {
        double loss = 0;
        for (int i = 0; i < predictions.size(); i++) {
            double diff = predictions.data()[i] - targets.data()[i];
            loss += diff * diff;
        }
        return loss / predictions.rows();
    }

    @Override
    public Tensor backward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int classes = predictions.cols();
        Tensor grad = Tensor.zeros(batchSize, classes);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < classes; j++) {
                double target = (j == labels[i]) ? 1.0 : 0.0;
                grad.data()[i * classes + j] = (float) (2.0 * (predictions.get(i, j) - target) / batchSize);
            }
        }
        return grad;
    }

    public Tensor backward(Tensor predictions, Tensor targets) {
        Tensor grad = predictions.subtract(targets).multiply(2.0 / predictions.rows());
        return grad;
    }
}
