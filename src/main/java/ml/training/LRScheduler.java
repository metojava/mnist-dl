package ml.training;

import ml.core.ThreadPoolManager;

import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class LRScheduler {
    public enum Policy {
        STEP_DECAY,
        EXPONENTIAL_DECAY,
        COSINE_ANNEALING,
        REDUCE_ON_PLATEAU,
        WARMUP_COSINE
    }

    private final Policy policy;
    private final ReentrantReadWriteLock lrLock = new ReentrantReadWriteLock();
    private volatile double baseLr;
    private volatile double currentLr;

    // Step decay params
    private int stepSize = 10;
    private double gamma = 0.1;

    // Cosine annealing params
    private int maxEpochs = 100;
    private double minLr = 1e-6;

    // Warmup params
    private int warmupEpochs = 5;

    // Reduce on plateau params
    private double patienceFactor = 0.5;
    private int patience = 5;
    private double bestLoss = Double.MAX_VALUE;
    private int epochsSinceImprovement = 0;

    public LRScheduler(Policy policy, double initialLr) {
        this.policy = policy;
        this.baseLr = initialLr;
        this.currentLr = initialLr;
    }

    public static LRScheduler stepDecay(double initialLr, int stepSize, double gamma) {
        LRScheduler s = new LRScheduler(Policy.STEP_DECAY, initialLr);
        s.stepSize = stepSize;
        s.gamma = gamma;
        return s;
    }

    public static LRScheduler exponentialDecay(double initialLr, double gamma) {
        LRScheduler s = new LRScheduler(Policy.EXPONENTIAL_DECAY, initialLr);
        s.gamma = gamma;
        return s;
    }

    public static LRScheduler cosineAnnealing(double initialLr, int maxEpochs, double minLr) {
        LRScheduler s = new LRScheduler(Policy.COSINE_ANNEALING, initialLr);
        s.maxEpochs = maxEpochs;
        s.minLr = minLr;
        return s;
    }

    public static LRScheduler warmupCosine(double initialLr, int maxEpochs, int warmupEpochs) {
        LRScheduler s = new LRScheduler(Policy.WARMUP_COSINE, initialLr);
        s.maxEpochs = maxEpochs;
        s.warmupEpochs = warmupEpochs;
        s.minLr = 1e-6;
        return s;
    }

    public static LRScheduler reduceOnPlateau(double initialLr, double factor, int patience) {
        LRScheduler s = new LRScheduler(Policy.REDUCE_ON_PLATEAU, initialLr);
        s.patienceFactor = factor;
        s.patience = patience;
        return s;
    }

    public double step(int epoch) {
        return step(epoch, 0);
    }

    public double step(int epoch, double currentLoss) {
        double newLr;
        switch (policy) {
            case STEP_DECAY -> newLr = baseLr * Math.pow(gamma, epoch / stepSize);
            case EXPONENTIAL_DECAY -> newLr = baseLr * Math.pow(gamma, epoch);
            case COSINE_ANNEALING -> {
                newLr = minLr + 0.5 * (baseLr - minLr) *
                    (1 + Math.cos(Math.PI * epoch / maxEpochs));
            }
            case WARMUP_COSINE -> {
                if (epoch < warmupEpochs) {
                    newLr = baseLr * ((double) (epoch + 1) / warmupEpochs);
                } else {
                    int cosineEpoch = epoch - warmupEpochs;
                    int cosineMax = maxEpochs - warmupEpochs;
                    newLr = minLr + 0.5 * (baseLr - minLr) *
                        (1 + Math.cos(Math.PI * cosineEpoch / cosineMax));
                }
            }
            case REDUCE_ON_PLATEAU -> {
                if (currentLoss < bestLoss) {
                    bestLoss = currentLoss;
                    epochsSinceImprovement = 0;
                } else {
                    epochsSinceImprovement++;
                }
                if (epochsSinceImprovement >= patience) {
                    newLr = currentLr * patienceFactor;
                    epochsSinceImprovement = 0;
                } else {
                    newLr = currentLr;
                }
            }
            default -> newLr = baseLr;
        }
        lrLock.writeLock().lock();
        try { currentLr = newLr; }
        finally { lrLock.writeLock().unlock(); }
        return newLr;
    }

    public double getCurrentLr() {
        lrLock.readLock().lock();
        try { return currentLr; }
        finally { lrLock.readLock().unlock(); }
    }

    public void setBaseLr(double lr) {
        lrLock.writeLock().lock();
        try { this.baseLr = lr; this.currentLr = lr; }
        finally { lrLock.writeLock().unlock(); }
    }

    // ==================== SCHEDULED AUTO-UPDATE ====================

    /**
     * Starts a background thread on the scheduler pool that calls `action`
     * with the updated learning rate at a fixed interval.
     */
    public ScheduledFuture<?> startAutoUpdate(long intervalMs,
                                               java.util.function.DoubleConsumer onLrChange) {
        ScheduledExecutorService scheduler = ThreadPoolManager.scheduler();
        DoubleAdder epochCounter = new DoubleAdder();
        return scheduler.scheduleAtFixedRate(() -> {
            double epoch = epochCounter.sum();
            double newLr = step((int) epoch);
            onLrChange.accept(newLr);
            epochCounter.add(1);
        }, intervalMs, intervalMs, TimeUnit.MILLISECONDS);
    }

    @Override
    public String toString() {
        return String.format("LRScheduler{policy=%s, lr=%.6f}", policy, getCurrentLr());
    }
}
