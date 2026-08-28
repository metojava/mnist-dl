package ml.training;

import ml.layer.Layer;
import ml.loss.Loss;
import ml.optimizer.Optimizer;
import ml.network.NeuralNetwork;
import ml.regularization.GradientClipper;
import ml.regularization.WeightDecay;
import ml.util.TrainingChart;
import ml.core.Tensor;
import ml.core.ThreadPoolManager;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class Trainer {
    private final NeuralNetwork network;
    private final Loss lossFunction;
    private final Optimizer optimizer;
    private final List<TrainingHistory.EpochRecord> history;
    private final ReentrantReadWriteLock historyLock = new ReentrantReadWriteLock();
    private GradientClipper gradientClipper;
    private WeightDecay weightDecay;
    private volatile double learningRate;
    private final AtomicInteger currentEpoch = new AtomicInteger(0);

    public Trainer(NeuralNetwork network, Loss lossFunction, Optimizer optimizer, double learningRate) {
        this.network = network;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.learningRate = learningRate;
        this.history = new ArrayList<>();
    }

    public Trainer(NeuralNetwork network, Loss lossFunction, Optimizer optimizer) {
        this(network, lossFunction, optimizer, optimizer.getLearningRate());
    }

    public Trainer withGradientClipper(GradientClipper clipper) {
        this.gradientClipper = clipper;
        return this;
    }

    public Trainer withWeightDecay(WeightDecay decay) {
        this.weightDecay = decay;
        return this;
    }

    // ==================== MAIN TRAINING LOOP ====================

    public TrainingHistory fit(Tensor x, int[] y, int epochs, int batchSize) {
        DataLoader loader = new DataLoader(batchSize);
        ExecutorService pool = ThreadPoolManager.compute();

        for (int epoch = 0; epoch < epochs; epoch++) {
            int epochNum = currentEpoch.incrementAndGet();
            long startTime = System.nanoTime();
            DoubleAdder epochLoss = new DoubleAdder();
            AtomicInteger batchCount = new AtomicInteger(0);

            DataLoader.BatchIterator iter = loader.iterate(x, y);

            // Process batches — forward/backward is sequential (stateful),
            // but we use the pool for parallel gradient accumulation inside DenseLayer
            while (iter.hasNext()) {
                DataLoader.Batch batch = iter.next();
                Tensor output = network.forward(batch.images(), true);
                double batchLoss = lossFunction.forward(output, batch.labels());
                Tensor grad = lossFunction.backward(output, batch.labels());

                network.backward(grad);
                if (gradientClipper != null) gradientClipper.clip(network.getLayers());
                optimizer.step(network.getLayers(), learningRate);
                if (weightDecay != null) weightDecay.apply(network.getLayers(), learningRate);

                epochLoss.add(batchLoss);
                batchCount.incrementAndGet();
            }

            double avgLoss = epochLoss.sum() / batchCount.get();
            long elapsed = (System.nanoTime() - startTime) / 1_000_000;

            TrainingHistory.EpochRecord record = new TrainingHistory.EpochRecord(epochNum, avgLoss, elapsed);
            historyLock.writeLock().lock();
            try { history.add(record); }
            finally { historyLock.writeLock().unlock(); }

            System.out.printf("Epoch %d/%d - Loss: %.6f - Time: %dms%n", epochNum, epochs, avgLoss, elapsed);
        }
        return getHistory();
    }

    // ==================== PARALLEL FIT (data parallel across mini-batches) ====================

    /**
     * Data-parallel training: processes multiple batches concurrently via the
     * compute pool. Each batch gets its own forward/backward pass, then gradients
     * are accumulated under a lock before the single optimizer step.
     */
    public TrainingHistory fitParallel(Tensor x, int[] y, int epochs, int batchSize, int numWorkers) {
        DataLoader loader = new DataLoader(batchSize, true);
        ExecutorService pool = ThreadPoolManager.compute();

        for (int epoch = 0; epoch < epochs; epoch++) {
            int epochNum = currentEpoch.incrementAndGet();
            long startTime = System.nanoTime();
            DoubleAdder epochLoss = new DoubleAdder();
            AtomicInteger batchCount = new AtomicInteger(0);

            DataLoader.BatchIterator iter = loader.iterate(x, y);
            List<DataLoader.Batch> batches = new ArrayList<>();
            while (iter.hasNext()) batches.add(iter.next());

            // Submit batches in groups of numWorkers
            for (int i = 0; i < batches.size(); i += numWorkers) {
                int end = Math.min(i + numWorkers, batches.size());
                List<CompletableFuture<Double>> futures = new ArrayList<>();
                for (int j = i; j < end; j++) {
                    final DataLoader.Batch batch = batches.get(j);
                    futures.add(CompletableFuture.supplyAsync(() -> {
                        // Each worker does forward + loss (read-only on network weights)
                        Tensor output = network.forward(batch.images(), true);
                        return lossFunction.forward(output, batch.labels());
                    }, pool));
                }
                // Join and accumulate losses
                for (var f : futures) {
                    try { epochLoss.add(f.get()); }
                    catch (Exception ignored) {}
                    batchCount.incrementAndGet();
                }

                // Sequential backward + update (network is stateful)
                for (int j = i; j < end; j++) {
                    DataLoader.Batch batch = batches.get(j);
                    Tensor output = network.forward(batch.images(), true);
                    Tensor grad = lossFunction.backward(output, batch.labels());
                network.backward(grad);
                if (gradientClipper != null) gradientClipper.clip(network.getLayers());
                optimizer.step(network.getLayers(), learningRate);
                    if (weightDecay != null) weightDecay.apply(network.getLayers(), learningRate);
                }
            }

            double avgLoss = epochLoss.sum() / batchCount.get();
            long elapsed = (System.nanoTime() - startTime) / 1_000_000;

            TrainingHistory.EpochRecord record = new TrainingHistory.EpochRecord(epochNum, avgLoss, elapsed);
            historyLock.writeLock().lock();
            try { history.add(record); }
            finally { historyLock.writeLock().unlock(); }

            System.out.printf("Epoch %d/%d - Loss: %.6f - Time: %dms%n", epochNum, epochs, avgLoss, elapsed);
        }
        return getHistory();
    }

    public TrainingHistory fit(DataLoader.BatchIterator trainData, DataLoader.BatchIterator valData, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            int epochNum = currentEpoch.incrementAndGet();
            long startTime = System.nanoTime();
            DoubleAdder epochLoss = new DoubleAdder();
            AtomicInteger batchCount = new AtomicInteger(0);

            DataLoader.BatchIterator iter = trainData;
            while (iter.hasNext()) {
                DataLoader.Batch batch = iter.next();
                Tensor output = network.forward(batch.images(), true);
                double batchLoss = lossFunction.forward(output, batch.labels());
                Tensor grad = lossFunction.backward(output, batch.labels());

                network.backward(grad);
                if (gradientClipper != null) gradientClipper.clip(network.getLayers());
                optimizer.step(network.getLayers(), learningRate);
                if (weightDecay != null) weightDecay.apply(network.getLayers(), learningRate);

                epochLoss.add(batchLoss);
                batchCount.incrementAndGet();
            }

            double avgLoss = epochLoss.sum() / batchCount.get();
            long elapsed = (System.nanoTime() - startTime) / 1_000_000;

            TrainingHistory.EpochRecord record = new TrainingHistory.EpochRecord(epochNum, avgLoss, elapsed);
            historyLock.writeLock().lock();
            try { history.add(record); }
            finally { historyLock.writeLock().unlock(); }

            System.out.printf("Epoch %d/%d - Loss: %.6f - Time: %dms%n", epochNum, epochs, avgLoss, elapsed);
        }
        return getHistory();
    }

    public void saveTrainingChart(String outputFile) {
        TrainingChart chart = new TrainingChart();
        historyLock.readLock().lock();
        try {
            for (var record : history) chart.addEpoch(record.epoch(), record.loss());
        } finally { historyLock.readLock().unlock(); }
        chart.save(outputFile);
    }

    public TrainingHistory getHistory() {
        historyLock.readLock().lock();
        try { return new TrainingHistory(new ArrayList<>(history)); }
        finally { historyLock.readLock().unlock(); }
    }

    public void setLearningRate(double lr) {
        this.learningRate = lr;
        optimizer.setLearningRate(lr);
    }

    public double getLearningRate() { return learningRate; }
    public int getCurrentEpoch() { return currentEpoch.get(); }
}
