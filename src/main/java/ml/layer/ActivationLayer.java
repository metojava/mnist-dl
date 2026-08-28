package ml.layer;

import ml.core.Tensor;

public abstract class ActivationLayer implements Layer {
    protected Tensor lastInput;
    protected Tensor lastOutput;

    @Override
    public Tensor forward(Tensor input, boolean training) {
        this.lastInput = input;
        this.lastOutput = computeForward(input);
        return lastOutput;
    }

    protected abstract Tensor computeForward(Tensor input);

    @Override
    public int inputSize() { return -1; }
    @Override
    public int outputSize() { return -1; }
    @Override
    public int parameterCount() { return 0; }
    @Override
    public void update(double learningRate) {}
}
