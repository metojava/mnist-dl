package ml.core;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public final class ThreadPoolManager {
    private static final ReentrantReadWriteLock poolLock = new ReentrantReadWriteLock();
    private static volatile ExecutorService computePool;
    private static volatile ExecutorService ioPool;
    private static volatile ScheduledExecutorService schedulerPool;
    private static final AtomicInteger poolCount = new AtomicInteger(0);

    private static final int DEFAULT_CORE = Runtime.getRuntime().availableProcessors();
    private static final int MAX_THREADS = Runtime.getRuntime().availableProcessors() * 2;
    private static final long KEEP_ALIVE_MS = 60_000L;

    private ThreadPoolManager() {}

    public static ExecutorService compute() {
        ExecutorService pool = computePool;
        if (pool == null || pool.isShutdown()) {
            poolLock.writeLock().lock();
            try {
                if (computePool == null || computePool.isShutdown()) {
                    computePool = new ThreadPoolExecutor(
                        DEFAULT_CORE, MAX_THREADS,
                        KEEP_ALIVE_MS, TimeUnit.MILLISECONDS,
                        new LinkedBlockingQueue<>(1024),
                        new ThreadFactory() {
                            private final AtomicInteger id = new AtomicInteger(0);
                            @Override public Thread newThread(Runnable r) {
                                Thread t = new Thread(r, "ml-compute-" + id.getAndIncrement());
                                t.setDaemon(true);
                                t.setPriority(Thread.NORM_PRIORITY);
                                return t;
                            }
                        },
                        new ThreadPoolExecutor.CallerRunsPolicy());
                }
                pool = computePool;
            } finally {
                poolLock.writeLock().unlock();
            }
        }
        return pool;
    }

    public static ExecutorService io() {
        ExecutorService pool = ioPool;
        if (pool == null || pool.isShutdown()) {
            poolLock.writeLock().lock();
            try {
                if (ioPool == null || ioPool.isShutdown()) {
                    ioPool = Executors.newFixedThreadPool(
                        Math.max(2, DEFAULT_CORE / 2),
                        new ThreadFactory() {
                            private final AtomicInteger id = new AtomicInteger(0);
                            @Override public Thread newThread(Runnable r) {
                                Thread t = new Thread(r, "ml-io-" + id.getAndIncrement());
                                t.setDaemon(true);
                                t.setPriority(Thread.MIN_PRIORITY);
                                return t;
                            }
                        });
                }
                pool = ioPool;
            } finally {
                poolLock.writeLock().unlock();
            }
        }
        return pool;
    }

    public static ScheduledExecutorService scheduler() {
        ScheduledExecutorService pool = schedulerPool;
        if (pool == null || pool.isShutdown()) {
            poolLock.writeLock().lock();
            try {
                if (schedulerPool == null || schedulerPool.isShutdown()) {
                    schedulerPool = Executors.newScheduledThreadPool(2,
                        new ThreadFactory() {
                            private final AtomicInteger id = new AtomicInteger(0);
                            @Override public Thread newThread(Runnable r) {
                                Thread t = new Thread(r, "ml-scheduler-" + id.getAndIncrement());
                                t.setDaemon(true);
                                return t;
                            }
                        });
                }
                pool = schedulerPool;
            } finally {
                poolLock.writeLock().unlock();
            }
        }
        return pool;
    }

    public static int computePoolSize() {
        ExecutorService pool = computePool;
        if (pool instanceof ThreadPoolExecutor tpe) {
            int size = tpe.getPoolSize();
            return size > 0 ? size : DEFAULT_CORE;
        }
        return DEFAULT_CORE;
    }

    public static int parallelism() { return DEFAULT_CORE; }

    // ==================== PARALLEL FOR ====================

    public static void parallelFor(int fromInclusive, int toExclusive,
                                   java.util.function.IntConsumer action) {
        int len = toExclusive - fromInclusive;
        if (len <= 0) return;
        Parallel.parallelRows(len, (a, b) -> {
            for (int i = a; i < b; i++) action.accept(fromInclusive + i);
        });
    }

    // ==================== SHUTDOWN ====================

    public static void shutdownAll() {
        poolLock.writeLock().lock();
        try {
            shutdownQuietly(computePool);
            shutdownQuietly(ioPool);
            shutdownQuietly(schedulerPool);
            computePool = null;
            ioPool = null;
            schedulerPool = null;
        } finally {
            poolLock.writeLock().unlock();
        }
    }

    private static void shutdownQuietly(ExecutorService pool) {
        if (pool != null && !pool.isShutdown()) {
            pool.shutdown();
            try { pool.awaitTermination(5, TimeUnit.SECONDS); }
            catch (InterruptedException ignored) { pool.shutdownNow(); }
        }
    }
}
