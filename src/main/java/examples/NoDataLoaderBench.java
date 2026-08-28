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

public class NoDataLoaderBench {
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

        System.out.println("=== Manual batches, no DataLoader ===\n");

        int batchSize = 128;
        int n = trainData.images().rows();
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;

        for (int epoch = 0; epoch < 5; epoch++) {
            long start = System.currentTimeMillis();
            double totalLoss = 0;
            int batchCount = 0;

            // shuffle
            for (int i = n - 1; i > 0; i--) {
                int j = (int)(Math.random() * (i + 1));
                int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
            }

            for (int bStart = 0; bStart < n; bStart += batchSize) {
                int bEnd = Math.min(bStart + batchSize, n);
                int bSize = bEnd - bStart;

                // manually build batch
                double[] imgData = new double[bSize * 784];
                int[] labels = new int[bSize];
                for (int i = 0; i < bSize; i++) {
                    int idx = perm[bStart + i];
                    System.arraycopy(trainData.images().data(), idx * 784, imgData, i * 784, 784);
                    labels[i] = trainData.labels()[idx];
                }
                Tensor batchImg = Tensor.fromFlat(imgData, bSize, 784);

                Tensor output = network.forward(batchImg, true);
                double l = loss.forward(output, labels);
                Tensor grad = loss.backward(output, labels);
                network.backward(grad);
                optimizer.step(network.getLayers(), 0.1);
                totalLoss += l;
                batchCount++;
            }
            System.out.printf("Epoch %d - Loss: %.6f - Time: %dms%n",
                epoch + 1, totalLoss / batchCount, System.currentTimeMillis() - start);
        }
    }
}
