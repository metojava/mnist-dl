package ml.loss;

import ml.core.Tensor;

public class CosineEmbeddingLoss implements Loss {
    private final double margin;

    public CosineEmbeddingLoss(double margin) {
        this.margin = margin;
    }

    public CosineEmbeddingLoss() {
        this(0.5);
    }

    public double margin() { return margin; }

    @Override
    public double forward(Tensor predictions, int[] labels) {
        double loss = 0;
        int batchSize = labels.length;
        for (int i = 0; i < batchSize; i++) {
            double norm = 0;
            int cols = predictions.cols();
            for (int j = 0; j < cols; j++) {
                double v = predictions.get(i, j);
                norm += v * v;
            }
            norm = Math.sqrt(norm) + 1e-12;
            double cosSim = predictions.get(i, 0) / norm;
            int target = labels[i];
            if (target == 1) {
                loss += 1.0 - cosSim;
            } else {
                loss += Math.max(0, cosSim - margin);
            }
        }
        return loss / batchSize;
    }

    public double forward(Tensor x1, Tensor x2, int[] labels) {
        double loss = 0;
        int batchSize = labels.length;
        int cols = x1.cols();
        for (int i = 0; i < batchSize; i++) {
            double dot = 0, norm1 = 0, norm2 = 0;
            for (int j = 0; j < cols; j++) {
                double a = x1.get(i, j);
                double b = x2.get(i, j);
                dot += a * b;
                norm1 += a * a;
                norm2 += b * b;
            }
            double cosSim = dot / (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-12);
            int target = labels[i];
            if (target == 1) {
                loss += 1.0 - cosSim;
            } else {
                loss += Math.max(0, cosSim - margin);
            }
        }
        return loss / batchSize;
    }

    @Override
    public Tensor backward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int cols = predictions.cols();
        Tensor grad = Tensor.zeros(batchSize, cols);
        for (int i = 0; i < batchSize; i++) {
            double norm = 0;
            for (int j = 0; j < cols; j++) {
                double v = predictions.get(i, j);
                norm += v * v;
            }
            norm = Math.sqrt(norm) + 1e-12;
            int target = labels[i];
            if (target == 1) {
                for (int j = 0; j < cols; j++) {
                    double v = predictions.get(i, j);
                    grad.data()[i * cols + j] = (float) (-(1.0 / norm - v * v / (norm * norm * norm)) / batchSize);
                }
            } else {
                double cosSim = predictions.get(i, 0) / norm;
                if (cosSim > margin) {
                    for (int j = 0; j < cols; j++) {
                        double v = predictions.get(i, j);
                        grad.data()[i * cols + j] = (float) ((1.0 / norm - v * v / (norm * norm * norm)) / batchSize);
                    }
                }
            }
        }
        return grad;
    }

    public Tensor backward(Tensor x1, Tensor x2, int[] labels) {
        int batchSize = labels.length;
        int cols = x1.cols();
        Tensor grad1 = Tensor.zeros(batchSize, cols);
        Tensor grad2 = Tensor.zeros(batchSize, cols);
        for (int i = 0; i < batchSize; i++) {
            double dot = 0, norm1 = 0, norm2 = 0;
            for (int j = 0; j < cols; j++) {
                double a = x1.get(i, j);
                double b = x2.get(i, j);
                dot += a * b;
                norm1 += a * a;
                norm2 += b * b;
            }
            double denom = Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-12;
            double cosSim = dot / denom;
            int target = labels[i];
            if (target == 1) {
                for (int j = 0; j < cols; j++) {
                    double a = x1.get(i, j);
                    double b = x2.get(i, j);
                    grad1.data()[i * cols + j] = (float) (-(b / denom - dot * a / (norm1 * denom)) / batchSize);
                    grad2.data()[i * cols + j] = (float) (-(a / denom - dot * b / (norm2 * denom)) / batchSize);
                }
            } else if (cosSim > margin) {
                for (int j = 0; j < cols; j++) {
                    double a = x1.get(i, j);
                    double b = x2.get(i, j);
                    grad1.data()[i * cols + j] = (float) ((b / denom - dot * a / (norm1 * denom)) / batchSize);
                    grad2.data()[i * cols + j] = (float) ((a / denom - dot * b / (norm2 * denom)) / batchSize);
                }
            }
        }
        return grad1;
    }
}
