package ml;

public interface Layer {

    Tensor forward(Tensor input);

    Tensor backward(Tensor gradient);

    void update(double learningRate);

    int inputSize();

    int outputSize();
}
