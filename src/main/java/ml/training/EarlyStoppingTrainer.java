package ml.training;

import ml.layer.Layer;
import ml.loss.Loss;
import ml.optimizer.Optimizer;
import ml.network.NeuralNetwork;
import ml.regularization.GradientClipper;
import ml.regularization.WeightDecay;
import ml.core.Tensor;
import ml.core.ThreadPoolManager;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class EarlyStoppingTrainer {
    private final NeuralNetwork network;
    private final Loss lossFunction;
    private final Optimizer optimizer;
    private final List<TrainingHistory.EpochRecord> history;
    private final ReentrantReadWriteLock historyLock = new ReentrantReadWriteLock();
    private final AtomicInteger currentEpoch = new AtomicInteger(0);
    private volatile double learningRate;
    private GradientClipper gradientClipper;
    private WeightDecay weightDecay;
    private LRScheduler lrScheduler;

    private final int patience;
    private final double minDelta;
    private double bestValLoss = Double.MAX_VALUE;
    private int epochsWithoutImprovement = 0;
    private NeuralNetwork bestModel;

    public EarlyStoppingTrainer(NeuralNetwork network, Loss loss, Optimizer opt,
                                 int patience, double minDelta) {
        this.network = network;
        this.lossFunction = loss;
        this.optimizer = opt;
        this.learningRate = opt.getLearningRate();
        this.patience = patience;
        this.minDelta = minDelta;
        this.history = new ArrayList<>();
    }

    public EarlyStoppingTrainer(NeuralNetwork network, Loss loss, Optimizer opt, int patience) {
        this(network, loss, opt, patience, 1e-4);
    }

    public EarlyStoppingTrainer withGradientClipper(GradientClipper c) { this.gradientClipper = c; return this; }
    public EarlyStoppingTrainer withWeightDecay(WeightDecay w) { this.weightDecay = w; return this; }
    public EarlyStoppingTrainer withLRScheduler(LRScheduler s) { this.lrScheduler = s; return this; }

    /**
     * Trains with early stopping based on validation loss.
     * Saves the best model weights and restores them at the end.
     */
    public TrainingHistory fit(Tensor trainX, int[] trainY, Tensor valX, int[] valY,
                                int maxEpochs, int batchSize) {
        DataLoader loader = new DataLoader(batchSize);
        bestModel = cloneNetwork(network);

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            int epochNum = currentEpoch.incrementAndGet();
            long startTime = System.nanoTime();

            // Training phase
            DoubleAdder trainLoss = new DoubleAdder();
            AtomicInteger batchCount = new AtomicInteger(0);
            DataLoader.BatchIterator iter = loader.iterate(trainX, trainY);
            while (iter.hasNext()) {
                DataLoader.Batch batch = iter.next();
                Tensor output = network.forward(batch.images(), true);
                double bl = lossFunction.forward(output, batch.labels());
                Tensor grad = lossFunction.backward(output, batch.labels());
                network.backward(grad);
                if (gradientClipper != null) gradientClipper.clip(network.getLayers());
                optimizer.step(network.getLayers(), learningRate);
                if (weightDecay != null) weightDecay.apply(network.getLayers(), learningRate);
                trainLoss.add(bl);
                batchCount.incrementAndGet();
            }

            // Validation phase
            Tensor valOutput = network.forward(valX);
            double valLoss = lossFunction.forward(valOutput, valY);
            double avgTrainLoss = trainLoss.sum() / batchCount.get();
            long elapsed = (System.nanoTime() - startTime) / 1_000_000;

            // LR scheduling
            if (lrScheduler != null) {
                learningRate = lrScheduler.step(epochNum, valLoss);
                optimizer.setLearningRate(learningRate);
            }

            // Early stopping check
            if (valLoss < bestValLoss - minDelta) {
                bestValLoss = valLoss;
                epochsWithoutImprovement = 0;
                bestModel = cloneNetwork(network);
            } else {
                epochsWithoutImprovement++;
            }

            TrainingHistory.EpochRecord record = new TrainingHistory.EpochRecord(epochNum, avgTrainLoss, elapsed);
            historyLock.writeLock().lock();
            try { history.add(record); }
            finally { historyLock.writeLock().unlock(); }

            System.out.printf("Epoch %d/%d - Train: %.6f - Val: %.6f - LR: %.6f - Time: %dms%s%n",
                epochNum, maxEpochs, avgTrainLoss, valLoss, learningRate, elapsed,
                epochsWithoutImprovement == 0 ? " [BEST]" : "");

            if (epochsWithoutImprovement >= patience) {
                System.out.printf("Early stopping at epoch %d (no improvement for %d epochs)%n",
                    epochNum, patience);
                break;
            }
        }

        // Restore best model
        restoreNetwork(network, bestModel);
        System.out.println("Restored best model (val loss=" + String.format("%.6f", bestValLoss) + ")");
        return getHistory();
    }

    private NeuralNetwork cloneNetwork(NeuralNetwork src) {
        NeuralNetwork copy = new NeuralNetwork();
        for (Layer layer : src.getLayers()) {
            if (layer instanceof ml.layer.DenseLayer dense) {
                ml.layer.DenseLayer newDense = new ml.layer.DenseLayer(dense.inputSize(), dense.outputSize());
                System.arraycopy(dense.weights().data(), 0, newDense.weights().data(), 0, dense.weights().size());
                System.arraycopy(dense.bias().data(), 0, newDense.bias().data(), 0, dense.bias().size());
                copy.add(newDense);
            } else {
                copy.add(layer);
            }
        }
        return copy;
    }

    private void restoreNetwork(NeuralNetwork target, NeuralNetwork source) {
        for (int i = 0; i < target.numberOfLayers(); i++) {
            if (target.getLayers().get(i) instanceof ml.layer.DenseLayer td &&
                source.getLayers().get(i) instanceof ml.layer.DenseLayer sd) {
                System.arraycopy(sd.weights().data(), 0, td.weights().data(), 0, sd.weights().size());
                System.arraycopy(sd.bias().data(), 0, td.bias().data(), 0, sd.bias().size());
            }
        }
    }

    public TrainingHistory getHistory() {
        historyLock.readLock().lock();
        try { return new TrainingHistory(new ArrayList<>(history)); }
        finally { historyLock.readLock().unlock(); }
    }
}
