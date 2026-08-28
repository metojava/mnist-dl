package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Softmax extends ActivationLayer {
    @Override
    protected Tensor computeForward(Tensor input) {
        int rows = input.rows();
        int cols = input.cols();
        Tensor output = new Tensor(rows, cols);
        double[] id = input.data();
        double[] od = output.data();
        for (int i = 0; i < rows; i++) {
            int offset = i * cols;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                double v = id[offset + j];
                if (v > max) max = v;
            }
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                double e = Math.exp(id[offset + j] - max);
                od[offset + j] = e;
                sum += e;
            }
            double invSum = 1.0 / sum;
            for (int j = 0; j < cols; j++)
                od[offset + j] *= invSum;
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        // CrossEntropyLoss.backward() already computes the combined
        // softmax + cross-entropy gradient: (predictions - one_hot) / batchSize.
        // Pass through directly — do NOT apply the Softmax Jacobian again.
        return gradient;
    }
}
