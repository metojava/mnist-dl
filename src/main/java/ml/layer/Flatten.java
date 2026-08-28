package ml.layer;

import ml.core.Tensor;

public class Flatten implements Layer {
    private int lastRows;
    private int lastCols;

    @Override
    public Tensor forward(Tensor input, boolean training) {
        lastRows = input.rows();
        lastCols = input.cols();
        return input;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        return gradient;
    }

    @Override
    public void update(double learningRate) {}

    @Override
    public int inputSize() { return -1; }
    @Override
    public int outputSize() { return -1; }
    @Override
    public int parameterCount() { return 0; }
}
