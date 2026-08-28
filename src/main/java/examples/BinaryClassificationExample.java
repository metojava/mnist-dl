package examples;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.BinaryCrossEntropyLoss;
import ml.evaluation.*;
import ml.regularization.Dropout;
import ml.regularization.GradientClipper;

public class BinaryClassificationExample {
    public static void main(String[] args) {
        SeedManager.setSeed(42);
        System.out.println("=== Binary Classification (XOR) ===\n");

        int samples = 2000;
        int features = 8;
        Tensor X = new Tensor(samples, features);
        int[] y = new int[samples];
        java.util.Random rng = new java.util.Random(42);

        for (int i = 0; i < samples; i++) {
            double[] vals = new double[features];
            for (int j = 0; j < features; j++) vals[j] = rng.nextGaussian();
            System.arraycopy(vals, 0, X.data(), i * features, features);
            y[i] = (vals[0] + vals[1] > 0) ? 1 : 0;
        }

        int split = (int) (samples * 0.8);
        Tensor trainX = X.sliceRows(0, split);
        int[] trainY = java.util.Arrays.copyOf(y, split);
        Tensor testX = X.sliceRows(split, samples);
        int[] testY = java.util.Arrays.copyOfRange(y, split, samples);

        NeuralNetwork network = new NeuralNetwork();
        network.add(new ml.layer.DenseLayer(features, 32));
        network.add(new ml.activation.ReLU());
        network.add(new ml.regularization.Dropout(0.1));
        network.add(new ml.layer.DenseLayer(32, 16));
        network.add(new ml.activation.ReLU());
        network.add(new ml.layer.DenseLayer(16, 1));
        network.add(new ml.activation.Sigmoid());
        System.out.println(network);

        BinaryCrossEntropyLoss loss = new BinaryCrossEntropyLoss();
        Adam optimizer = new Adam(0.001);
        Trainer trainer = new Trainer(network, loss, optimizer);

        System.out.println("Training...");
        TrainingHistory history = trainer.fit(trainX, trainY, 30, 64);
        trainer.saveTrainingChart("training_binary.png");

        Evaluator evaluator = new Evaluator(new String[]{"Class 0", "Class 1"});
        EvaluationResult result = evaluator.evaluate(network, testX, testY);

        System.out.println("\n" + result);
        result.confusionMatrix().print();
        result.confusionMatrix().printClassMetrics(new String[]{"Negative", "Positive"});
    }
}
