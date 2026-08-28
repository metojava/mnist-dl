package ml.loss;

import ml.core.Tensor;

public class HingeLoss implements Loss {
    @Override
    public double forward(Tensor predictions, int[] labels) {
        double loss = 0;
        int batchSize = labels.length;
        int classes = predictions.cols();
        for (int i = 0; i < batchSize; i++) {
            int label = labels[i];
            double correctScore = predictions.get(i, label);
            for (int j = 0; j < classes; j++) {
                if (j != label) {
                    loss += Math.max(0, predictions.get(i, j) - correctScore + 1);
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
            int label = labels[i];
            double correctScore = predictions.get(i, label);
            for (int j = 0; j < classes; j++) {
                if (j != label) {
                    if (predictions.get(i, j) - correctScore + 1 > 0) {
                        grad.data()[i * classes + j] += 1.0 / batchSize;
                        grad.data()[i * classes + label] -= 1.0 / batchSize;
                    }
                }
            }
        }
        return grad;
    }
}
