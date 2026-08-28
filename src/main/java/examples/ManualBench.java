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

public class ManualBench {
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

        System.out.println("=== Manual training loop (no Trainer class) ===\n");

        DataLoader loader = new DataLoader(128);

        for (int epoch = 0; epoch < 5; epoch++) {
            long start = System.currentTimeMillis();
            double totalLoss = 0;
            int batchCount = 0;

            DataLoader.BatchIterator iter = loader.iterate(trainData.images(), trainData.labels());
            while (iter.hasNext()) {
                DataLoader.Batch batch = iter.next();
                Tensor output = network.forward(batch.images(), true);
                double batchLoss = loss.forward(output, batch.labels());
                Tensor grad = loss.backward(output, batch.labels());
                network.backward(grad);
                optimizer.step(network.getLayers(), 0.1);
                totalLoss += batchLoss;
                batchCount++;
            }
            long elapsed = System.currentTimeMillis() - start;
            System.out.printf("Epoch %d - Loss: %.6f - Time: %dms%n",
                epoch + 1, totalLoss / batchCount, elapsed);
        }
    }
}
