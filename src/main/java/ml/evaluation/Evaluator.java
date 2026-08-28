package ml.evaluation;

import ml.core.Tensor;
import ml.core.ThreadPoolManager;
import ml.network.NeuralNetwork;

import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class Evaluator {
    private final String[] classNames;

    public Evaluator(String[] classNames) {
        this.classNames = classNames;
    }

    public Evaluator() {
        this(null);
    }

    // ==================== PARALLEL EVALUATION ====================

    /**
     * Evaluates the network on the full dataset by splitting into batches
     * and processing them concurrently via the compute ExecutorService.
     * Uses a lock-free ConfusionMatrix with atomic-style accumulation.
     */
    public EvaluationResult evaluate(NeuralNetwork network, Tensor images, int[] labels) {
        int numClasses = classNames != null ? classNames.length : detectClasses(labels);
        int total = labels.length;
        int parallelism = ThreadPoolManager.parallelism();
        int batchSize = Math.max(1, total / (parallelism * 4));

        if (total <= batchSize * 2 || parallelism <= 1) {
            return evaluateSequential(network, images, labels, numClasses);
        }

        ExecutorService pool = ThreadPoolManager.compute();
        int numBatches = (total + batchSize - 1) / batchSize;
        ConfusionMatrix[] partialCMs = new ConfusionMatrix[numBatches];
        CompletableFuture<?>[] futures = new CompletableFuture[numBatches];

        for (int b = 0; b < numBatches; b++) {
            final int from = b * batchSize;
            final int to = Math.min(from + batchSize, total);
            final int idx = b;
            partialCMs[b] = new ConfusionMatrix(numClasses);

            futures[b] = CompletableFuture.runAsync(() -> {
                Tensor batchInput = images.sliceRows(from, to);
                int[] batchLabels = java.util.Arrays.copyOfRange(labels, from, to);
                int[] predictions = network.predictBatch(batchInput);
                for (int i = 0; i < predictions.length; i++)
                    partialCMs[idx].add(batchLabels[i], predictions[i]);
            }, pool);
        }

        CompletableFuture.allOf(futures).join();

        // Merge partial confusion matrices
        ConfusionMatrix merged = new ConfusionMatrix(numClasses);
        for (ConfusionMatrix cm : partialCMs) {
            int[][] arr = cm.toArray();
            for (int i = 0; i < numClasses; i++)
                for (int j = 0; j < numClasses; j++)
                    for (int k = 0; k < arr[i][j]; k++)
                        merged.add(i, j);
        }

        return new EvaluationResult(total, merged.correct(), merged);
    }

    private EvaluationResult evaluateSequential(NeuralNetwork network, Tensor images,
                                                 int[] labels, int numClasses) {
        int[] predictions = network.predictBatch(images);
        ConfusionMatrix cm = new ConfusionMatrix(numClasses);
        cm.addAll(labels, predictions);
        return new EvaluationResult(labels.length, cm.correct(), cm);
    }

    // ==================== PARALLEL PREDICTION ====================

    public PredictionResult predict(NeuralNetwork network, Tensor images, int[] labels) {
        int total = images.rows();
        int parallelism = ThreadPoolManager.parallelism();
        int batchSize = Math.max(1, total / (parallelism * 4));

        if (total <= batchSize * 2 || parallelism <= 1) {
            return predictSequential(network, images, labels);
        }

        ExecutorService pool = ThreadPoolManager.compute();
        int numBatches = (total + batchSize - 1) / batchSize;
        Prediction[][] partialResults = new Prediction[numBatches][];
        CompletableFuture<?>[] futures = new CompletableFuture[numBatches];

        for (int b = 0; b < numBatches; b++) {
            final int from = b * batchSize;
            final int to = Math.min(from + batchSize, total);
            final int idx = b;
            futures[b] = CompletableFuture.runAsync(() -> {
                Tensor batchInput = images.sliceRows(from, to);
                int[] batchLabels = java.util.Arrays.copyOfRange(labels, from, to);
                Tensor output = network.forward(batchInput);
                Prediction[] preds = new Prediction[to - from];
                for (int i = 0; i < preds.length; i++) {
                    int pred = output.argmaxRow(i);
                    double confidence = output.get(i, pred);
                    preds[i] = new Prediction(pred, batchLabels[i], confidence);
                }
                partialResults[idx] = preds;
            }, pool);
        }

        CompletableFuture.allOf(futures).join();

        Prediction[] all = new Prediction[total];
        for (int b = 0; b < numBatches; b++)
            System.arraycopy(partialResults[b], 0, all, b * batchSize, partialResults[b].length);

        return new PredictionResult(all);
    }

    private PredictionResult predictSequential(NeuralNetwork network, Tensor images, int[] labels) {
        Tensor output = network.forward(images);
        Prediction[] preds = new Prediction[images.rows()];
        for (int i = 0; i < images.rows(); i++) {
            int pred = output.argmaxRow(i);
            double confidence = output.get(i, pred);
            preds[i] = new Prediction(pred, labels[i], confidence);
        }
        return new PredictionResult(preds);
    }

    private int detectClasses(int[] labels) {
        int max = 0;
        for (int l : labels) if (l > max) max = l;
        return max + 1;
    }
}
