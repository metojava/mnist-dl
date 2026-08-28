package ml.layer;

import ml.core.Tensor;

public interface Layer {
    Tensor forward(Tensor input, boolean training);
    Tensor backward(Tensor gradient);
    void update(double learningRate);
    int inputSize();
    int outputSize();
    int parameterCount();
    default Tensor forward(Tensor input) {
        return forward(input, false);
    }
}
