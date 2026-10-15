package ml.activation;

import ml.core.Tensor;
import ml.layer.ActivationLayer;

public class Softmax extends ActivationLayer {
    private Tensor outputBuffer;

    @Override
    protected Tensor computeForward(Tensor input) {
        int rows = input.rows();
        int cols = input.cols();
        if (outputBuffer == null || outputBuffer.rows() != rows || outputBuffer.cols() != cols)
            outputBuffer = new Tensor(rows, cols);

        float[] id = input.data();
        float[] od = outputBuffer.data();
        for (int i = 0; i < rows; i++) {
            int offset = i * cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                float v = id[offset + j];
                if (v > max) max = v;
            }
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                float e = (float) Math.exp(id[offset + j] - max);
                od[offset + j] = e;
                sum += e;
            }
            float invSum = 1.0f / sum;
            for (int j = 0; j < cols; j++)
                od[offset + j] *= invSum;
        }
        return outputBuffer;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        // CrossEntropyLoss.backward() already computes the combined
        // softmax + cross-entropy gradient: (predictions - one_hot) / batchSize.
        // Pass through directly - do NOT apply the Softmax Jacobian again.
        return gradient;
    }
}
