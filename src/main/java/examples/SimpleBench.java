package examples;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.SGD;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import ml.training.Dataset;
import ml.util.MNISTLoader;

public class SimpleBench {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);

        System.out.println("=== Simple 2-Layer (Keras-style) Fashion-MNIST ===\n");

        Dataset trainData = MNISTLoader.load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
        Dataset testData = MNISTLoader.load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

        // Simple: Dense(784→128) + ReLU + Dense(128→10) + Softmax
        NeuralNetwork network = new NeuralNetwork();
        network.add(new ml.layer.DenseLayer(784, 128));
        network.add(new ml.activation.ReLU());
        network.add(new ml.layer.DenseLayer(128, 10));
        network.add(new ml.activation.Softmax());

        System.out.println(network);

        SGD optimizer = new SGD(0.1);
        CrossEntropyLoss loss = new CrossEntropyLoss();
        Trainer trainer = new Trainer(network, loss, optimizer);

        System.out.println("Training 10 epochs, batch=128...\n");
        long start = System.currentTimeMillis();
        trainer.fit(trainData.images(), trainData.labels(), 10, 128);
        long elapsed = System.currentTimeMillis() - start;
        System.out.printf("\nTraining: %dms (%.0f ms/epoch)%n", elapsed, elapsed / 10.0);

        // Evaluate
        System.out.println("\nEvaluating...");
        int correct = 0;
        int total = testData.images().rows();
        int batchSize = 256;
        for (int i = 0; i < total; i += batchSize) {
            int end = Math.min(i + batchSize, total);
            Tensor batch = testData.images().sliceRows(i, end);
            int[] preds = network.predictBatch(batch);
            int[] labels = java.util.Arrays.copyOfRange(testData.labels(), i, end);
            for (int j = 0; j < preds.length; j++) {
                if (preds[j] == labels[j]) correct++;
            }
        }
        System.out.printf("Accuracy: %d/%d = %.2f%%%n", correct, total, 100.0 * correct / total);
    }
}
