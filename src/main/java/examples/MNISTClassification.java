package examples;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.optimizer.SGD;
import ml.loss.CrossEntropyLoss;
import ml.loss.MSELoss;
import ml.evaluation.*;
import ml.regularization.Dropout;
import ml.regularization.GradientClipper;
import ml.regularization.WeightDecay;
import ml.util.MNISTLoader;
import ml.util.MNISTVisualizer;
import ml.util.TrainingChart;

public class MNISTClassification {
    private static final String[] CLASS_NAMES = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);

        System.out.println("=== Fashion-MNIST Classification ===\n");

        long loadStart = System.currentTimeMillis();
        Dataset trainData = MNISTLoader.load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
        Dataset testData = MNISTLoader.load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
        long loadTime = System.currentTimeMillis() - loadStart;

        System.out.println("\nBuilding network with Adam optimizer...");
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
        System.out.println(network);

        Adam optimizer = new Adam(0.001);
        CrossEntropyLoss loss = new CrossEntropyLoss();
        Trainer trainer = new Trainer(network, loss, optimizer)
            .withGradientClipper(new GradientClipper(5.0))
            .withWeightDecay(new WeightDecay(0, 1e-4));

        System.out.println("Training...");
        long trainStart = System.currentTimeMillis();
        TrainingHistory history = trainer.fit(trainData.images(), trainData.labels(), 10, 256);
        long trainTime = System.currentTimeMillis() - trainStart;

        trainer.saveTrainingChart("training_adam.png");

        System.out.println("\nEvaluating...");
        long evalStart = System.currentTimeMillis();
        Evaluator evaluator = new Evaluator(CLASS_NAMES);
        EvaluationResult result = evaluator.evaluate(network, testData.images(), testData.labels());
        long evalTime = System.currentTimeMillis() - evalStart;

        System.out.println("\n" + result);
        System.out.println();
        result.confusionMatrix().print();
        System.out.println();
        result.confusionMatrix().printClassMetrics(CLASS_NAMES);

        System.out.println("\nGenerating prediction visualization...");
        PredictionResult predResult = evaluator.predict(network,
            testData.images().sliceRows(0, 20), java.util.Arrays.copyOf(testData.labels(), 20));
        double[] confidences = new double[20];
        int[] predicted = new int[20];
        int[] actual = new int[20];
        for (int i = 0; i < 20; i++) {
            predicted[i] = predResult.get(i).predictedLabel();
            actual[i] = predResult.get(i).actualLabel();
            confidences[i] = predResult.get(i).confidence();
        }
        new MNISTVisualizer(CLASS_NAMES).visualize(
            testData.images().sliceRows(0, 20), predicted, actual, confidences, 20, "predictions_adam.png");

        System.out.println("\n=== Summary ===");
        System.out.printf("Data loading:  %dms%n", loadTime);
        System.out.printf("Training:      %dms%n", trainTime);
        System.out.printf("Evaluation:    %dms%n", evalTime);
        System.out.printf("Total time:    %dms%n", loadTime + trainTime + evalTime);
        System.out.printf("Parameters:    %d%n", network.totalParameters());

        System.out.println("\nSaving model...");
        network.saveModel("model_fashion_mnist.bin");
    }
}
