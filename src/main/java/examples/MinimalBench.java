package examples;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.optimizer.SGD;
import ml.loss.CrossEntropyLoss;
import ml.layer.DenseLayer;
import ml.layer.Layer;
import ml.util.MNISTLoader;
import ml.training.Dataset;

public class MinimalBench {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        Dataset trainData = MNISTLoader.load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");

        NeuralNetwork network = new NeuralNetwork();
        network.add(new DenseLayer(784, 128));
        network.add(new ml.activation.ReLU());
        network.add(new DenseLayer(128, 10));
        network.add(new ml.activation.Softmax());

        SGD optimizer = new SGD(0.1);
        CrossEntropyLoss loss = new CrossEntropyLoss();

        Tensor fixedBatch = trainData.images().sliceRows(0, 128);
        int[] fixedLabels = java.util.Arrays.copyOfRange(trainData.labels(), 0, 128);

        System.out.println("=== Fixed-batch training: same 128 samples every step ===\n");

        for (int epoch = 0; epoch < 5; epoch++) {
            double totalLoss = 0;
            long start = System.currentTimeMillis();
            for (int step = 0; step < 100; step++) {
                Tensor output = network.forward(fixedBatch, true);
                double l = loss.forward(output, fixedLabels);
                Tensor grad = loss.backward(output, fixedLabels);
                network.backward(grad);
                optimizer.step(network.getLayers(), 0.1);
                totalLoss += l;
            }
            System.out.printf("Epoch %d - AvgLoss(100 steps): %.6f - Time: %dms%n",
                epoch + 1, totalLoss / 100, System.currentTimeMillis() - start);
        }
    }
}
