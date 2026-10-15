package projects.vision;

import ml.core.Tensor;
import projects.common.Proj;
import ml.core.SeedManager;
import projects.common.Proj;
import ml.network.NeuralNetwork;
import projects.common.Proj;
import ml.training.*;
import projects.common.Proj;
import ml.optimizer.Adam;
import projects.common.Proj;
import ml.loss.CrossEntropyLoss;
import projects.common.Proj;
import ml.regularization.Dropout;
import projects.common.Proj;

import java.util.Random;
import projects.common.Proj;

/**
 * P05 - Ensemble (bagging) of three MNIST networks with majority voting.
 * Demonstrates that averaging diverse weak learners beats each individual model.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.vision.MnistEnsemble" "-Dexec.args=6 256"
 */
public class MnistEnsemble {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 6), batch = Proj.batch(args, 256);
        long t0 = System.currentTimeMillis();

        Dataset train = ml.util.MNISTLoader.load(
            "data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte");
        Dataset test = ml.util.MNISTLoader.load(
            "data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte");

        // Bootstrap subsets for diversity
        Random rng = new Random(42);
        int n = train.size();
        Tensor[] bootX = new Tensor[3];
        int[][] bootY = new int[3][];
        for (int m = 0; m < 3; m++) {
            float[] d = new float[n * 784];
            int[] y = new int[n];
            for (int i = 0; i < n; i++) {
                int j = rng.nextInt(n);
                System.arraycopy(train.images().data(), j * 784, d, i * 784, 784);
                y[i] = train.labels()[j];
            }
            bootX[m] = Tensor.fromFlat(d, n, 784);
            bootY[m] = y;
        }

        NeuralNetwork[] nets = new NeuralNetwork[3];
        double[] accs = new double[3];
        for (int m = 0; m < 3; m++) {
            nets[m] = buildNet(m);
            new Trainer(nets[m], new CrossEntropyLoss(), new Adam(0.001))
                .fit(bootX[m], bootY[m], epochs, batch);
            accs[m] = accuracy(nets[m], test.images(), test.labels());
            System.out.printf("Model %d accuracy: %.2f%%%n", m, accs[m] * 100);
        }

        // Majority vote
        Tensor[] outs = new Tensor[3];
        for (int m = 0; m < 3; m++) outs[m] = nets[m].forward(test.images(), false);
        int correct = 0;
        for (int i = 0; i < test.size(); i++) {
            int[] votes = new int[10];
            for (int m = 0; m < 3; m++) votes[outs[m].argmaxRow(i)]++;
            int best = 0;
            for (int k = 1; k < 10; k++) if (votes[k] > votes[best]) best = k;
            if (best == test.labels()[i]) correct++;
        }
        double ens = 100.0 * correct / test.size();
        System.out.printf("%nEnsemble accuracy: %.2f%%  (best single: %.2f%%)%n", ens, java.util.Arrays.stream(accs).max().orElse(0) * 100);
        Proj.footer("P05 MnistEnsemble", t0);
    }

    private static NeuralNetwork buildNet(int seed) {
        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(784, 256));
        net.add(new ml.activation.ReLU());
        net.add(new Dropout(0.25));
        net.add(new ml.layer.DenseLayer(256, 64));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(64, 10));
        net.add(new ml.activation.Softmax());
        return net;
    }

    static double accuracy(NeuralNetwork net, Tensor x, int[] y) {
        Tensor out = net.forward(x, false);
        int c = 0;
        for (int i = 0; i < out.rows(); i++) if (out.argmaxRow(i) == y[i]) c++;
        return (double) c / x.rows();
    }
}


