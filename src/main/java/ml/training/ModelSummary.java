package ml.training;

import ml.network.NeuralNetwork;
import ml.layer.Layer;

public class ModelSummary {
    public static String summary(NeuralNetwork network) {
        StringBuilder sb = new StringBuilder();
        sb.append("Model: NeuralNetwork\n");
        sb.append("================================================================\n");
        sb.append(String.format("%-4s %-25s %-15s %-15s %-10s\n",
            "#", "Layer (type)", "Output Shape", "Params", "Connected to"));
        sb.append("================================================================\n");

        int totalParams = 0;
        int totalTrainable = 0;

        for (int i = 0; i < network.numberOfLayers(); i++) {
            Layer layer = network.getLayers().get(i);
            int params = layer.parameterCount();
            totalParams += params;
            totalTrainable += params;

            String layerName = layer.getClass().getSimpleName();
            String outputShape = layer.outputSize() > 0 ? "(*" + layer.outputSize() + ")" : "(*)";
            String connectedTo = i > 0 ? "[" + (i - 1) + "]" : "input";

            sb.append(String.format("%-4d %-25s %-15s %-15s %-10s\n",
                i, layerName, outputShape, params, connectedTo));
        }

        sb.append("================================================================\n");
        sb.append(String.format("Total params: %,d\n", totalParams));
        sb.append(String.format("Trainable params: %,d\n", totalTrainable));
        sb.append("Non-trainable params: 0\n");
        sb.append("================================================================\n");

        return sb.toString();
    }

    public static void printSummary(NeuralNetwork network) {
        System.out.println(summary(network));
    }

    public static int countParams(NeuralNetwork network) {
        return network.totalParameters();
    }

    public static long estimateMemoryBytes(NeuralNetwork network) {
        long bytes = 0;
        for (Layer layer : network.getLayers()) {
            bytes += (long) layer.parameterCount() * 8;
        }
        return bytes;
    }

    public static String memorySummary(NeuralNetwork network) {
        long bytes = estimateMemoryBytes(network);
        return String.format("Estimated model memory: %,d bytes (%.2f MB)", bytes, bytes / (1024.0 * 1024.0));
    }
}
