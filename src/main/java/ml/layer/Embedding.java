package ml.layer;

import ml.core.Tensor;
import java.util.Random;

public class Embedding implements Layer {
    private final int vocabSize;
    private final int embeddingDim;
    private final Tensor weights;
    private int lastIndices[];
    private double[] weightGrads;

    public Embedding(int vocabSize, int embeddingDim, Random rng) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.weights = Tensor.xavier(vocabSize, embeddingDim, rng);
    }

    public Embedding(int vocabSize, int embeddingDim) {
        this(vocabSize, embeddingDim, new Random());
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int batchSize = input.rows();
        Tensor output = Tensor.zeros(batchSize, embeddingDim);
        lastIndices = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            int idx = (int) input.data()[i];
            if (idx < 0 || idx >= vocabSize)
                throw new IndexOutOfBoundsException("Embedding index " + idx + " out of range [0, " + vocabSize + ")");
            lastIndices[i] = idx;
            System.arraycopy(weights.data(), idx * embeddingDim, output.data(), i * embeddingDim, embeddingDim);
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        if (weightGrads == null) weightGrads = new double[vocabSize * embeddingDim];
        for (int i = 0; i < lastIndices.length; i++) {
            int idx = lastIndices[i];
            for (int j = 0; j < embeddingDim; j++) {
                weightGrads[idx * embeddingDim + j] += gradient.data()[i * embeddingDim + j];
            }
        }
        return gradient;
    }

    @Override
    public void update(double learningRate) {
        if (weightGrads == null) return;
        for (int i = 0; i < weightGrads.length; i++) {
            weights.data()[i] -= learningRate * weightGrads[i];
            weightGrads[i] = 0.0;
        }
    }

    @Override
    public int inputSize() { return 1; }
    @Override
    public int outputSize() { return embeddingDim; }
    @Override
    public int parameterCount() { return vocabSize * embeddingDim; }

    public Tensor weights() { return weights; }
    public int vocabSize() { return vocabSize; }
    public int embeddingDim() { return embeddingDim; }
}
