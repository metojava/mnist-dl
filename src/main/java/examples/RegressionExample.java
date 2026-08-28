package examples;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.MSELoss;
import ml.evaluation.Evaluator;
import ml.regularization.GradientClipper;

public class RegressionExample {
    public static void main(String[] args) {
        SeedManager.setSeed(42);
        System.out.println("=== Simple Regression Example ===\n");

        int samples = 1000;
        int features = 5;
        Tensor X = new Tensor(samples, features);
        double[] trueWeights = {2.0, -1.5, 0.5, 3.0, -0.8};
        double trueBias = 1.5;
        double[] y = new double[samples];
        java.util.Random rng = new java.util.Random(42);

        for (int i = 0; i < samples; i++) {
            double sum = trueBias;
            for (int j = 0; j < features; j++) {
                double val = rng.nextGaussian();
                X.data()[i * features + j] = val;
                sum += val * trueWeights[j];
            }
            y[i] = sum + rng.nextGaussian() * 0.1;
        }

        System.out.println("True weights: [2.0, -1.5, 0.5, 3.0, -0.8], bias: 1.5\n");

        NeuralNetwork network = new NeuralNetwork();
        network.add(new ml.layer.DenseLayer(features, 32));
        network.add(new ml.activation.ReLU());
        network.add(new ml.layer.DenseLayer(32, 16));
        network.add(new ml.activation.ReLU());
        network.add(new ml.layer.DenseLayer(16, 1));
        System.out.println(network);

        MSELoss loss = new MSELoss();
        Adam optimizer = new Adam(0.01);
        Trainer trainer = new Trainer(network, loss, optimizer)
            .withGradientClipper(new GradientClipper(1.0));

        TrainingHistory history = trainer.fit(X, yToIntArray(y), 50, 64);
        trainer.saveTrainingChart("training_regression.png");

        System.out.println("\nFinal loss: " + history.lastLoss());
        System.out.println("Best loss:  " + history.bestLoss());
        System.out.println("Total time: " + history.totalTimeMs() + "ms");
    }

    private static int[] yToIntArray(double[] y) {
        int[] result = new int[y.length];
        for (int i = 0; i < y.length; i++) result[i] = (int) y[i];
        return result;
    }
}
