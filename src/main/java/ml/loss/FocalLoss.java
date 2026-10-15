package ml.loss;

import ml.core.Tensor;

public class FocalLoss implements Loss {
    private final double alpha;
    private final double gamma;
    private final double epsilon;

    public FocalLoss(double alpha, double gamma, double epsilon) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
    }

    public FocalLoss(double alpha, double gamma) {
        this(alpha, gamma, 1e-12);
    }

    public FocalLoss() {
        this(0.25, 2.0);
    }

    public double alpha() { return alpha; }
    public double gamma() { return gamma; }

    @Override
    public double forward(Tensor predictions, int[] labels) {
        double loss = 0;
        int batchSize = labels.length;
        int classes = predictions.cols();
        for (int i = 0; i < batchSize; i++) {
            int label = labels[i];
            double p = Math.max(predictions.get(i, label), epsilon);
            double focalWeight = alpha * Math.pow(1.0 - p, gamma);
            loss -= focalWeight * Math.log(p);
        }
        return loss / batchSize;
    }

    @Override
    public Tensor backward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int classes = predictions.cols();
        double scale = 1.0 / batchSize;
        Tensor grad = new Tensor(batchSize, classes);
        float[] pd = predictions.data();
        float[] gd = grad.data();
        for (int i = 0; i < batchSize; i++) {
            int label = labels[i];
            for (int j = 0; j < classes; j++) {
                double p = Math.max(pd[i * classes + j], epsilon);
                if (j == label) {
                    double focalWeight = alpha * Math.pow(1.0 - p, gamma);
                    double dFocal = -gamma * alpha * Math.pow(1.0 - p, gamma - 1) * Math.log(p)
                                    - focalWeight / p;
                    gd[i * classes + j] = (float) (dFocal * scale);
                } else {
                    double focalWeight = alpha * Math.pow(1.0 - p, gamma);
                    double dFocal = -gamma * alpha * Math.pow(1.0 - p, gamma - 1) * Math.log(p)
                                    - focalWeight / p;
                    gd[i * classes + j] = 0.0f;
                }
            }
        }
        return grad;
    }
}
