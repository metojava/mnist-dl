package ml.core;

import java.util.concurrent.ThreadLocalRandom;
import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;

public final class SeedManager {
    private static volatile long globalSeed = -1;
    private static final ReentrantLock seedLock = new ReentrantLock();
    private static final ThreadLocal<Random> threadLocalRandom = ThreadLocal.withInitial(
        () -> globalSeed >= 0 ? new Random(globalSeed + Thread.currentThread().getId()) : new Random()
    );

    private SeedManager() {}

    public static void setSeed(long seed) {
        seedLock.lock();
        try {
            globalSeed = seed;
            threadLocalRandom.remove();
        } finally {
            seedLock.unlock();
        }
    }

    public static long getSeed() {
        return globalSeed;
    }

    public static boolean isSeeded() {
        return globalSeed >= 0;
    }

    public static Random getRandom() {
        return threadLocalRandom.get();
    }

    public static Random newRandom() {
        if (globalSeed >= 0) {
            return new Random(globalSeed + Thread.currentThread().getId());
        }
        return new Random();
    }

    public static ThreadLocalRandom threadLocal() {
        return ThreadLocalRandom.current();
    }

    public static void reset() {
        seedLock.lock();
        try {
            globalSeed = -1;
            threadLocalRandom.remove();
        } finally {
            seedLock.unlock();
        }
    }
}
