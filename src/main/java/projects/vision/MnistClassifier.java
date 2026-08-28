package projects.vision;

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
import ml.evaluation.*;
import projects.common.Proj;
import ml.regularization.Dropout;
import projects.common.Proj;
import ml.regularization.GradientClipper;
import projects.common.Proj;
import ml.regularization.WeightDecay;
import projects.common.Proj;
import ml.util.MNISTLoader;
import projects.common.Proj;
import ml.util.MNISTVisualizer;
import projects.common.Proj;

import java.nio.file.Files;
import projects.common.Proj;
import java.nio.file.Path;
import projects.common.Proj;

/**
 * P01 - Handwritten digit classification (MNIST, 60k/10k, 28x28 grayscale).
 * Data : data/mnist/*.idx3-ubyte
 * Run  : mvn compile exec:java "-Dexec.mainClass=projects.vision.MnistClassifier" "-Dexec.args=12 256"
 */
public class MnistClassifier {
    private static final String[] CLASS_NAMES = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 12), batch = Proj.batch(args, 256);
        Path out = Proj.out("mnist");
        long t0 = System.currentTimeMillis();

        Dataset train = MNISTLoader.load("data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte");
        Dataset test = MNISTLoader.load("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte");

        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(784, 512));
        net.add(new ml.activation.ReLU());
        net.add(new Dropout(0.2));
        net.add(new ml.layer.DenseLayer(512, 256));
        net.add(new ml.activation.ReLU());
        net.add(new Dropout(0.2));
        net.add(new ml.layer.DenseLayer(256, 128));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(128, 10));
        net.add(new ml.activation.Softmax());
        System.out.printf("[P01 MNIST] params=%d epochs=%d batch=%d%n", net.totalParameters(), epochs, batch);

        Trainer trainer = new Trainer(net, new CrossEntropyLoss(), new Adam(0.001))
            .withGradientClipper(new GradientClipper(5.0))
            .withWeightDecay(new WeightDecay(0, 1e-4));
        TrainingHistory history = trainer.fit(train.images(), train.labels(), epochs, batch);
        trainer.saveTrainingChart(out.resolve("training_loss.png").toString());

        EvaluationResult result = new Evaluator(CLASS_NAMES).evaluate(net, test.images(), test.labels());
        System.out.println(result);
        result.confusionMatrix().printClassMetrics(CLASS_NAMES);

        PredictionResult pr = evaluator(net, test);
        new MNISTVisualizer(CLASS_NAMES).visualize(test.images().sliceRows(0, 20),
            labels(pr, 20, true), labels(pr, 20, false), confidences(pr, 20), 20,
            out.resolve("predictions.png").toString());

        net.saveModel(out.resolve("model.bin").toString());
        System.out.printf("Accuracy: %.2f%% | artifacts: %s%n", result.accuracy() * 100, out.toAbsolutePath());
        Proj.footer("P01 MNIST", t0);
    }

    private static PredictionResult evaluator(NeuralNetwork net, Dataset test) {
        return new Evaluator(CLASS_NAMES).predict(net, test.images().sliceRows(0, 20),
            java.util.Arrays.copyOf(test.labels(), 20));
    }
    private static int[] labels(PredictionResult pr, int n, boolean predicted) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = predicted ? pr.get(i).predictedLabel() : pr.get(i).actualLabel();
        return a;
    }
    private static double[] confidences(PredictionResult pr, int n) {
        double[] c = new double[n];
        for (int i = 0; i < n; i++) c[i] = pr.get(i).confidence();
        return c;
    }
}

