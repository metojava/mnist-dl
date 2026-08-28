package examples;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.SGD;
import ml.loss.CrossEntropyLoss;
import ml.layer.DenseLayer;
import ml.layer.Layer;
import ml.util.MNISTLoader;

public class DiagBench {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);

        Dataset trainData = MNISTLoader.load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");

        NeuralNetwork network = new NeuralNetwork();
        network.add(new ml.layer.DenseLayer(784, 128));
        network.add(new ml.activation.ReLU());
        network.add(new ml.layer.DenseLayer(128, 10));
        network.add(new ml.activation.Softmax());

        SGD optimizer = new SGD(0.1);
        CrossEntropyLoss loss = new CrossEntropyLoss();

        System.out.println("=== DIAGNOSTIC: 1 batch, manual forward/backward/update ===\n");

        // Take first batch manually
        Tensor xBatch = trainData.images().sliceRows(0, 32);
        int[] yBatch = java.util.Arrays.copyOfRange(trainData.labels(), 0, 32);

        System.out.println("Batch input: mean=" + xBatch.mean() + " sum=" + xBatch.sum());
        System.out.println("Batch labels: " + java.util.Arrays.toString(yBatch));

        // Forward
        Tensor output = network.forward(xBatch, true);
        System.out.println("\nForward output (before softmax): checking shape " + output.rows() + "x" + output.cols());
        System.out.println("Output row 0: ");
        double[] row0 = output.data();
        for (int j = 0; j < Math.min(10, output.cols()); j++) {
            System.out.printf("  [%d]=%.6f", j, row0[j]);
        }
        System.out.println();
        System.out.println("Output sum row0 = " + output.sum());
        System.out.println("Output max = " + output.max() + " min = " + output.min());

        // Loss
        double batchLoss = loss.forward(output, yBatch);
        System.out.println("\nLoss: " + batchLoss);

        // Backward
        Tensor grad = loss.backward(output, yBatch);
        System.out.println("\nLoss gradient: sum=" + grad.sum() + " max=" + grad.max() + " min=" + grad.min());

        // Check gradient stats
        double[] gd = grad.data();
        int nonZero = 0;
        for (double v : gd) if (v != 0) nonZero++;
        System.out.println("Loss gradient non-zero elements: " + nonZero + "/" + gd.length);

        network.backward(grad);

        // Check weight gradients
        for (Layer layer : network.getLayers()) {
            if (layer instanceof DenseLayer dense) {
                Tensor wg = dense.weightsGradient();
                Tensor bg = dense.biasGradient();
                if (wg != null) {
                    System.out.printf("\nDense(%d->%d) weightGrad: sum=%.6f max=%.6f min=%.6f norm=%.6f%n",
                        dense.inputSize(), dense.outputSize(),
                        wg.sum(), wg.max(), wg.min(),
                        Math.sqrt(wg.data().length > 0 ? wg.data()[0] * wg.data()[0] : 0));
                    double wNorm = 0;
                    for (double v : wg.data()) wNorm += v * v;
                    System.out.printf("  weightGrad L2 = %.6f%n", Math.sqrt(wNorm));
                }
                if (bg != null) {
                    double bNorm = 0;
                    for (double v : bg.data()) bNorm += v * v;
                    System.out.printf("  biasGrad L2 = %.6f%n", Math.sqrt(bNorm));
                }
            }
        }

        // Check weights before update
        System.out.println("\n=== BEFORE UPDATE ===");
        for (Layer layer : network.getLayers()) {
            if (layer instanceof DenseLayer dense) {
                double[] wData = dense.weights().data();
                double wMean = 0;
                for (double v : wData) wMean += v;
                wMean /= wData.length;
                double wNorm = 0;
                for (double v : wData) wNorm += v * v;
                System.out.printf("Dense(%d->%d) weights: mean=%.6f L2=%.6f%n",
                    dense.inputSize(), dense.outputSize(), wMean, Math.sqrt(wNorm));
            }
        }

        // Step
        optimizer.step(network.getLayers(), 0.1);

        // Check weights after update
        System.out.println("\n=== AFTER UPDATE ===");
        for (Layer layer : network.getLayers()) {
            if (layer instanceof DenseLayer dense) {
                double[] wData = dense.weights().data();
                double wMean = 0;
                for (double v : wData) wMean += v;
                wMean /= wData.length;
                double wNorm = 0;
                for (double v : wData) wNorm += v * v;
                System.out.printf("Dense(%d->%d) weights: mean=%.6f L2=%.6f%n",
                    dense.inputSize(), dense.outputSize(), wMean, Math.sqrt(wNorm));
            }
        }

        // Second forward to verify loss changed
        Tensor output2 = network.forward(xBatch, true);
        double batchLoss2 = loss.forward(output2, yBatch);
        System.out.println("\nLoss after 1 step: " + batchLoss2);
        System.out.println("Loss delta: " + (batchLoss - batchLoss2));
    }
}
