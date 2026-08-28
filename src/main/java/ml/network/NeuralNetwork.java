package ml.network;

import ml.layer.Layer;
import ml.core.Tensor;
import ml.exception.LayerConfigurationException;
import ml.exception.ModelIOException;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers;

    public NeuralNetwork() {
        this.layers = new ArrayList<>();
    }

    public NeuralNetwork add(Layer layer) {
        if (!layers.isEmpty()) {
            Layer prev = layers.get(layers.size() - 1);
            if (prev.outputSize() > 0 && layer.inputSize() > 0) {
                if (prev.outputSize() != layer.inputSize()) {
                    throw new LayerConfigurationException(
                        String.format("Dimension mismatch: layer %d outputs %d, but layer %d expects %d",
                            layers.size() - 1, prev.outputSize(), layers.size(), layer.inputSize()));
                }
            }
        }
        layers.add(layer);
        return this;
    }

    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (Layer layer : layers) current = layer.forward(current);
        return current;
    }

    public Tensor forward(Tensor input, boolean training) {
        Tensor current = input;
        for (Layer layer : layers) current = layer.forward(current, training);
        return current;
    }

    public void backward(Tensor gradient) {
        Tensor current = gradient;
        for (int i = layers.size() - 1; i >= 0; i--)
            current = layers.get(i).backward(current);
    }

    public void update(double learningRate) {
        for (Layer layer : layers) layer.update(learningRate);
    }

    public int predict(Tensor input) {
        Tensor output = forward(input);
        return output.argmaxRow(0);
    }

    public int[] predictBatch(Tensor input) {
        Tensor output = forward(input);
        int[] predictions = new int[output.rows()];
        for (int i = 0; i < output.rows(); i++)
            predictions[i] = output.argmaxRow(i);
        return predictions;
    }

    public double[] predictSingle(double[] input) {
        Tensor t = new Tensor(new double[][]{input});
        Tensor output = forward(t);
        return output.toArray()[0];
    }

    public int totalParameters() {
        int total = 0;
        for (Layer layer : layers) total += layer.parameterCount();
        return total;
    }

    // ==================== MODEL SAVE/LOAD ====================

    public void saveModel(String filepath) throws ModelIOException {
        ModelSerializer.saveBinary(this, filepath);
    }

    public static NeuralNetwork loadModel(String filepath) throws ModelIOException {
        return ModelSerializer.loadBinary(filepath);
    }

    public void saveJson(String filepath) throws ModelIOException {
        ModelSerializer.saveJson(this, filepath);
    }

    public static NeuralNetwork loadJson(String filepath) throws ModelIOException {
        return ModelSerializer.loadJson(filepath);
    }

    public void saveSafetensors(String filepath) throws ModelIOException {
        ModelSerializer.saveSafetensors(this, filepath);
    }

    public static NeuralNetwork loadSafetensors(String filepath) throws ModelIOException {
        return ModelSerializer.loadSafetensors(filepath);
    }

    public void saveOnnx(String filepath) throws ModelIOException {
        ModelSerializer.saveOnnx(this, filepath);
    }

    public static NeuralNetwork loadOnnx(String filepath) throws ModelIOException {
        return ModelSerializer.loadOnnx(filepath);
    }

    public List<Layer> getLayers() { return layers; }
    public int numberOfLayers() { return layers.size(); }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("NeuralNetwork[\n");
        int totalParams = 0;
        for (int i = 0; i < layers.size(); i++) {
            Layer l = layers.get(i);
            int params = l.parameterCount();
            totalParams += params;
            sb.append(String.format("  [%d] %s (in=%d, out=%d, params=%d)%n",
                i, l.getClass().getSimpleName(), l.inputSize(), l.outputSize(), params));
        }
        sb.append(String.format("  Total parameters: %d%n", totalParams));
        sb.append("]");
        return sb.toString();
    }
}
