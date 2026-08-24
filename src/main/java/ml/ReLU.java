package ml;

public class ReLU implements Activation {

    private Tensor input;

    @Override
    public Tensor forward(Tensor input) {

        this.input = input;

        Tensor output =
                new Tensor(input.rows(), input.cols());

        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) {

                output.set(
                        i,
                        j,
                        Math.max(0.0, input.get(i, j))
                );
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {

        Tensor result =
                new Tensor(
                        gradient.rows(),
                        gradient.cols()
                );

        for (int i = 0; i < gradient.rows(); i++) {
            for (int j = 0; j < gradient.cols(); j++) {

                double derivative =
                        input.get(i, j) > 0 ? 1.0 : 0.0;

                result.set(
                        i,
                        j,
                        gradient.get(i, j) * derivative
                );
            }
        }

        return result;
    }

    @Override
    public void update(double learningRate) {
        // ReLU has no trainable parameters.
    }

    @Override
    public int inputSize() {
        return -1;
    }

    @Override
    public int outputSize() {
        return -1;
    }
}
