package ml;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final List<Layer> layers =
            new ArrayList<>();

    public NeuralNetwork add(Layer layer) {

        /*
         * Check dimensions against the previous
         * trainable layer where possible.
         */
        if (!layers.isEmpty()) {

            Layer previous =
                    layers.get(layers.size() - 1);

            int previousOutput =
                    previous.outputSize();

            int currentInput =
                    layer.inputSize();

            if (previousOutput > 0 &&
                    currentInput > 0 &&
                    previousOutput != currentInput) {

                throw new IllegalArgumentException(
                        "Layer dimensions don't match: " +
                        previousOutput +
                        " -> " +
                        currentInput
                );
            }
        }

        layers.add(layer);

        return this;
    }

    public Tensor forward(Tensor input) {

        Tensor result = input;

        for (Layer layer : layers) {
            result = layer.forward(result);
        }

        return result;
    }

    public Tensor backward(Tensor gradient) {

        Tensor result = gradient;

        for (int i = layers.size() - 1; i >= 0; i--) {

            result =
                    layers.get(i).backward(result);
        }

        return result;
    }

    public void update(double learningRate) {

        for (Layer layer : layers) {
            layer.update(learningRate);
        }
    }

    public int predict(Tensor input) {

        Tensor output = forward(input);

        return output.argmaxRow(0);
    }

    public int predictSingle(float[] input) {

        Tensor tensor =
                new Tensor(
                        new float[][] { input }
                );

        return predict(tensor);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public int numberOfLayers() {
        return layers.size();
    }
}
