package projects.vision;

import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import projects.common.Proj;
import ml.optimizer.AdamW;
import ml.loss.CrossEntropyLoss;
import ml.evaluation.*;
import ml.regularization.Dropout;

/**
 * P02 - Fashion-MNIST clothing classification (T-shirt...Ankle boot).
 * Data : data/fashion/*-idx*-ubyte
 * Run  : mvn compile exec:java "-Dexec.mainClass=projects.vision.FashionMnistClassifier" "-Dexec.args=15 256"
 */
public class FashionMnistClassifier {
    private static final String[] CLASS_NAMES = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 15), batch = Proj.batch(args, 256);
        var out = Proj.out("fashion");
        long t0 = System.currentTimeMillis();

        Dataset train = ml.util.MNISTLoader.load(
            "data/fashion/train-images-idx3-ubyte", "data/fashion/train-labels-idx1-ubyte");
        Dataset test = ml.util.MNISTLoader.load(
            "data/fashion/t10k-images-idx3-ubyte", "data/fashion/t10k-labels-idx1-ubyte");

        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(784, 400));
        net.add(new ml.activation.ReLU());
        net.add(new Dropout(0.25));
        net.add(new ml.layer.DenseLayer(400, 192));
        net.add(new ml.activation.ReLU());
        net.add(new Dropout(0.25));
        net.add(new ml.layer.DenseLayer(192, 10));
        net.add(new ml.activation.Softmax());
        System.out.printf("[P02 Fashion-MNIST] params=%d epochs=%d batch=%d%n", net.totalParameters(), epochs, batch);

        Trainer trainer = new Trainer(net, new CrossEntropyLoss(), new AdamW(0.0015));
        trainer.fit(train.images(), train.labels(), epochs, batch);
        trainer.saveTrainingChart(out.resolve("training_loss.png").toString());

        EvaluationResult result = new Evaluator(CLASS_NAMES).evaluate(net, test.images(), test.labels());
        System.out.println(result);
        result.confusionMatrix().printClassMetrics(CLASS_NAMES);
        net.saveModel(out.resolve("model.bin").toString());
        System.out.printf("Accuracy: %.2f%% | artifacts: %s%n", result.accuracy() * 100, out.toAbsolutePath());
        Proj.footer("P02 Fashion-MNIST", t0);
    }
}


