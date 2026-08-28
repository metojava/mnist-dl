package ml.loss;

import ml.core.Tensor;

public class CrossEntropyLoss implements Loss {
    private final double epsilon;

    public CrossEntropyLoss(double epsilon) {
        this.epsilon = epsilon;
    }

    public CrossEntropyLoss() {
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

    @Override
    public Tensor backward(Tensor predictions, int[] labels) {
        int batchSize = labels.length;
        int classes = predictions.cols();
        double scale = 1.0 / batchSize;
        Tensor grad = new Tensor(batchSize, classes);
        double[] pd = predictions.data();
        double[] gd = grad.data();
        System.arraycopy(pd, 0, gd, 0, pd.length);
        for (int i = 0; i < batchSize; i++) {
            gd[i * classes + labels[i]] -= 1.0;
        }
        for (int i = 0; i < gd.length; i++) gd[i] *= scale;
        return grad;
    }
}
