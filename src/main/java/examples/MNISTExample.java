package examples;

import ml.CrossEntropyLoss;
import ml.DenseLayer;
import ml.EvaluationResult;
import ml.Evaluator;
import ml.MNISTLoader;
import ml.MNISTVisualizer;
import ml.NeuralNetwork;
import ml.ReLU;
import ml.SGD;
import ml.Softmax;
import ml.Trainer;

public class MNISTExample {

    /*
     * Fashion-MNIST class labels.
     *
     * The numeric labels are:
     *
     * 0 = T-shirt/top
     * 1 = Trouser
     * 2 = Pullover
     * 3 = Dress
     * 4 = Coat
     * 5 = Sandal
     * 6 = Shirt
     * 7 = Sneaker
     * 8 = Bag
     * 9 = Ankle boot
     */
    public static final String[] FASHION_LABELS = {
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
    };

    public static void main(String[] args)
            throws Exception {

        // =========================================================
        // Configuration
        // =========================================================

        String trainImages =
                "data/train-images-idx3-ubyte";

        String trainLabels =
                "data/train-labels-idx1-ubyte";

        String testImages =
                "data/t10k-images-idx3-ubyte";

        String testLabels =
                "data/t10k-labels-idx1-ubyte";

        /*
         * Training configuration.
         *
         * Fashion-MNIST contains:
         *
         * 60,000 training images
         * 10,000 test images
         *
         * Each image is 28 x 28 = 784 pixels.
         */
        int epochs = 10;

        if (args.length > 0) {
            epochs = Integer.parseInt(args[0]);
        }

        int batchSize = 512;

        double learningRate = 0.01;

        /*
         * Number of test images to put into predictions.png.
         */
        int numberOfPredictions = 20;

        // =========================================================
        // Start
        // =========================================================

        long programStart =
                System.nanoTime();

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "          JAVA FASHION-MNIST DEEP LEARNING"
        );

        System.out.println(
                "=================================================="
        );

        System.out.println();

        // =========================================================
        // Dataset information
        // =========================================================

        System.out.println(
                "Dataset: Fashion-MNIST"
        );

        System.out.println(
                "Classes: 10"
        );

        System.out.println(
                "Image size: 28 x 28"
        );

        System.out.println(
                "Input size: 784"
        );

        System.out.println();

        System.out.println(
                "Fashion-MNIST classes:"
        );

        for (int i = 0;
             i < FASHION_LABELS.length;
             i++) {

            System.out.printf(
                    "  %d = %s%n",
                    i,
                    FASHION_LABELS[i]
            );
        }

        // =========================================================
        // Load Fashion-MNIST
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "[1/6] Loading Fashion-MNIST dataset..."
        );

        System.out.println(
                "=================================================="
        );

        long loadingStart =
                System.nanoTime();

        MNISTLoader.Dataset train =
                MNISTLoader.load(
                        trainImages,
                        trainLabels
                );

        MNISTLoader.Dataset test =
                MNISTLoader.load(
                        testImages,
                        testLabels
                );

        long loadingEnd =
                System.nanoTime();

        double loadingSeconds =
                (loadingEnd - loadingStart)
                        / 1_000_000_000.0;

        System.out.printf(
                "%nDataset loading time: %.2f seconds%n",
                loadingSeconds
        );

        System.out.println();

        System.out.println(
                "Training samples: "
                        + train.images().rows()
        );

        System.out.println(
                "Test samples:     "
                        + test.images().rows()
        );

        System.out.println(
                "Input features:   "
                        + train.images().cols()
        );

        // =========================================================
        // Build neural network
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "[2/6] Building neural network..."
        );

        System.out.println(
                "=================================================="
        );

        NeuralNetwork network =
                new NeuralNetwork()

                        /*
                         * 784 input pixels
                         *
                         * 28 x 28 = 784
                         */
                        .add(
                                new DenseLayer(
                                        784,
                                        128
                                )
                        )

                        /*
                         * ReLU activation
                         */
                        .add(
                                new ReLU()
                        )

                        /*
                         * Hidden layer
                         */
                        .add(
                                new DenseLayer(
                                        128,
                                        64
                                )
                        )

                        /*
                         * ReLU activation
                         */
                        .add(
                                new ReLU()
                        )

                        /*
                         * 10 output classes
                         */
                        .add(
                                new DenseLayer(
                                        64,
                                        10
                                )
                        )

                        /*
                         * Convert logits to probabilities.
                         */
                        .add(
                                new Softmax()
                        );

        System.out.println();

        System.out.println(
                "Network architecture:"
        );

        System.out.println(
                "  Input:      784"
        );

        System.out.println(
                "  Dense:      784 -> 128"
        );

        System.out.println(
                "  Activation: ReLU"
        );

        System.out.println(
                "  Dense:      128 -> 64"
        );

        System.out.println(
                "  Activation: ReLU"
        );

        System.out.println(
                "  Dense:      64 -> 10"
        );

        System.out.println(
                "  Output:     Softmax"
        );

        System.out.println();

        System.out.println(
                "Output classes:"
        );

        for (int i = 0;
             i < FASHION_LABELS.length;
             i++) {

            System.out.printf(
                    "  %d -> %s%n",
                    i,
                    FASHION_LABELS[i]
            );
        }

        // =========================================================
        // Loss and optimizer
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "[3/6] Configuring training..."
        );

        System.out.println(
                "=================================================="
        );

        SGD optimizer =
                new SGD(
                        learningRate
                );

        CrossEntropyLoss loss =
                new CrossEntropyLoss();

        System.out.printf(
                "Epochs:        %d%n",
                epochs
        );

        System.out.printf(
                "Batch size:    %d%n",
                batchSize
        );

        System.out.printf(
                "Learning rate: %.5f%n",
                learningRate
        );

        // =========================================================
        // Train
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "[4/6] Training Fashion-MNIST network..."
        );

        System.out.println(
                "=================================================="
        );

        System.out.println();

        long trainingStart =
                System.nanoTime();

        Trainer trainer =
                new Trainer(
                        network,
                        optimizer,
                        loss
                );

        trainer.fit(
                train.images(),
                train.labels(),
                epochs,
                batchSize
        );

        long trainingEnd =
                System.nanoTime();

        double trainingSeconds =
                (trainingEnd - trainingStart)
                        / 1_000_000_000.0;

        System.out.println();

        System.out.printf(
                "Training completed in %.2f seconds%n",
                trainingSeconds
        );

        System.out.printf(
                "Average time per epoch: %.2f seconds%n",
                trainingSeconds / epochs
        );

        // =========================================================
        // Save training chart
        // =========================================================

        System.out.println();

        System.out.println(
                "Generating training chart..."
        );

        String trainingChart =
                "training.png";

        trainer.saveTrainingChart(
                trainingChart
        );

        System.out.println(
                "Training chart saved to: "
                        + trainingChart
        );

        // =========================================================
        // Evaluate
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "[5/6] Evaluating Fashion-MNIST test set..."
        );

        System.out.println(
                "=================================================="
        );

        long evaluationStart =
                System.nanoTime();

        EvaluationResult result =
                Evaluator.evaluate(
                        network,
                        test.images(),
                        test.labels()
                );

        long evaluationEnd =
                System.nanoTime();

        double evaluationSeconds =
                (evaluationEnd - evaluationStart)
                        / 1_000_000_000.0;

        // =========================================================
        // Results
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "              TEST RESULTS"
        );

        System.out.println(
                "=================================================="
        );

        System.out.printf(
                "Dataset:       Fashion-MNIST%n"
        );

        System.out.printf(
                "Total samples: %d%n",
                result.total()
        );

        System.out.printf(
                "Correct:       %d%n",
                result.correct()
        );

        System.out.printf(
                "Incorrect:     %d%n",
                result.incorrect()
        );

        System.out.printf(
                "Accuracy:      %.2f%%%n",
                result.accuracyPercentage()
        );

        System.out.printf(
                "Evaluation:    %.2f seconds%n",
                evaluationSeconds
        );

        System.out.println(
                "=================================================="
        );

        // =========================================================
        // Confusion matrix
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "              CONFUSION MATRIX"
        );

        System.out.println(
                "=================================================="
        );

        result
                .confusionMatrix()
                .print();

        // =========================================================
        // Per-class metrics
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "              CLASS METRICS"
        );

        System.out.println(
                "=================================================="
        );

        result
                .confusionMatrix()
                .printClassMetrics();

        // =========================================================
        // Prediction visualization
        // =========================================================

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "[6/6] Generating predictions visualization..."
        );

        System.out.println(
                "=================================================="
        );

        String predictionsImage =
                "predictions.png";

        MNISTVisualizer.savePredictions(
                network,
                test.images(),
                test.labels(),
                0,
                numberOfPredictions,
                predictionsImage
        );

        System.out.println();

        System.out.println(
                "Prediction image saved to: "
                        + predictionsImage
        );

        // =========================================================
        // Program timing
        // =========================================================

        long programEnd =
                System.nanoTime();

        double programSeconds =
                (programEnd - programStart)
                        / 1_000_000_000.0;

        System.out.println();

        System.out.println(
                "=================================================="
        );

        System.out.println(
                "                  COMPLETE"
        );

        System.out.println(
                "=================================================="
        );

        System.out.println();

        System.out.printf(
                "Fashion-MNIST accuracy: %.2f%%%n",
                result.accuracyPercentage()
        );

        System.out.printf(
                "Dataset loading:        %.2f sec%n",
                loadingSeconds
        );

        System.out.printf(
                "Training:               %.2f sec%n",
                trainingSeconds
        );

        System.out.printf(
                "Evaluation:             %.2f sec%n",
                evaluationSeconds
        );

        System.out.printf(
                "Total program time:     %.2f sec%n",
                programSeconds
        );

        System.out.println();

        System.out.println(
                "Generated files:"
        );

        System.out.println(
                "  - training.png"
        );

        System.out.println(
                "  - predictions.png"
        );

        System.out.println();

        System.out.println(
                "Fashion-MNIST labels:"
        );

        for (int i = 0;
             i < FASHION_LABELS.length;
             i++) {

            System.out.printf(
                    "  %d = %s%n",
                    i,
                    FASHION_LABELS[i]
            );
        }

        System.out.println();
    }
}
