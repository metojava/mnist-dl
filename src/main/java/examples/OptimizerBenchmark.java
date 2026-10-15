package examples;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.*;
import ml.loss.*;
import ml.evaluation.*;
import ml.regularization.*;
import ml.activation.*;
import java.util.Random;

public class OptimizerBenchmark {
    public static void main(String[] args) {
        SeedManager.setSeed(42);
        System.out.println("=== Optimizer Benchmark ===\n");

        int samples = 2000;
        int features = 20;
        int classes = 5;
        Random rng = new Random(42);

        Tensor X = new Tensor(samples, features);
        int[] y = new int[samples];
        for (int i = 0; i < samples; i++) {
            int label = rng.nextInt(classes);
            y[i] = label;
            for (int j = 0; j < features; j++)
                X.data()[i * features + j] = (float) (rng.nextGaussian() + label * 0.5);
        }

        int split = (int) (samples * 0.8);
        Tensor trainX = X.sliceRows(0, split);
        int[] trainY = java.util.Arrays.copyOf(y, split);
        Tensor testX = X.sliceRows(split, samples);
        int[] testY = java.util.Arrays.copyOfRange(y, split, samples);

        Optimizer[] optimizers = {
            new SGD(0.01),
            new SGD(0.01, 0.9),
            new Adam(0.001),
            new AdamW(0.001, 0.01),
            new RMSProp(0.001),
            new AdaGrad(0.01)
        };

        for (Optimizer opt : optimizers) {
            System.out.println("--- " + opt.getClass().getSimpleName() + " ---");
            NeuralNetwork net = new NeuralNetwork();
            net.add(new ml.layer.DenseLayer(features, 64));
            net.add(new ReLU());
            net.add(new ml.layer.DenseLayer(64, 32));
            net.add(new ReLU());
            net.add(new ml.layer.DenseLayer(32, classes));
            net.add(new Softmax());

            CrossEntropyLoss loss = new CrossEntropyLoss();
            Trainer trainer = new Trainer(net, loss, opt);
            TrainingHistory history = trainer.fit(trainX, trainY, 20, 64);

            Evaluator evaluator = new Evaluator();
            EvaluationResult result = evaluator.evaluate(net, testX, testY);
            System.out.printf("  Final loss: %.6f | Accuracy: %.2f%% | Time: %dms%n%n",
                history.lastLoss(), result.accuracyPercentage(), history.totalTimeMs());
        }
    }
}
