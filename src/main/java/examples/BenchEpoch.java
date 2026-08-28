package examples;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import ml.util.MNISTLoader;

public class BenchEpoch {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        Dataset trainData = MNISTLoader.load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");

        NeuralNetwork network = new NeuralNetwork();
        network.add(new ml.layer.DenseLayer(784, 256));
        network.add(new ml.activation.ReLU());
        network.add(new ml.regularization.Dropout(0.2));
        network.add(new ml.layer.DenseLayer(256, 128));
        network.add(new ml.activation.ReLU());
        network.add(new ml.regularization.Dropout(0.2));
        network.add(new ml.layer.DenseLayer(128, 64));
        network.add(new ml.activation.ReLU());
        network.add(new ml.layer.DenseLayer(64, 10));
        network.add(new ml.activation.Softmax());

        Adam optimizer = new Adam(0.001);
        CrossEntropyLoss loss = new CrossEntropyLoss();
        Trainer trainer = new Trainer(network, loss, optimizer)
            .withGradientClipper(new ml.regularization.GradientClipper(5.0))
            .withWeightDecay(new ml.regularization.WeightDecay(0, 1e-4));

        System.out.println("Warming up...");
        trainer.fit(trainData.images(), trainData.labels(), 1, 256);

        System.out.println("\nBenchmarking 3 epochs...");
        long start = System.currentTimeMillis();
        trainer.fit(trainData.images(), trainData.labels(), 3, 256);
        long elapsed = System.currentTimeMillis() - start;
        System.out.printf("%nTotal: %dms for 3 epochs => %.0f ms/epoch%n", elapsed, elapsed / 3.0);
    }
}
