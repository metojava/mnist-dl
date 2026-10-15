package ml.core;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Minimal row-parallel helper used by the GEMM and element-wise kernels.
 *
 * A single fixed-size daemon pool is shared by all kernels. Each task
 * partitions the requested range into one contiguous slice per worker,
 * so threads only ever write disjoint output elements and no locking is
 * needed between accumulations.
 *
 * Crucially, a call made from inside the pool runs its slice inline
 * instead of submitting nested work and blocking. This prevents the
 * classic nested-submission deadlock that occurs when worker threads
 * wait on tasks queued to the same bounded executor.
 */
public final class Parallel {

    @FunctionalInterface
    public interface RangeTask {
        void run(int from, int to);
    }

    private static final int THREADS =
            Math.max(1, Math.min(8, Runtime.getRuntime().availableProcessors()));

    private static final ExecutorService POOL =
            THREADS <= 1
                    ? null
                    : Executors.newFixedThreadPool(THREADS, runnable -> {
                        Thread t = new Thread(runnable, "ml-parallel");
                        t.setDaemon(true);
                        return t;
                    });

    private static final ThreadLocal<Boolean> IN_POOL =
            ThreadLocal.withInitial(() -> Boolean.FALSE);

    private Parallel() {
    }

    public static int threads() {
        return THREADS;
    }

    /**
     * Runs {@code task} over [0, totalRows), split into contiguous slices.
     * Blocks until all slices complete. If called from a pool thread, the
     * task runs inline to avoid nested thread-pool submission.
     */
    public static void parallelRows(int totalRows, RangeTask task) {
        if (THREADS <= 1 || totalRows <= 1 || IN_POOL.get()) {
            task.run(0, totalRows);
            return;
        }

        int slices = Math.min(THREADS, totalRows);
        int sliceSize = (totalRows + slices - 1) / slices;
        CountDownLatch latch = new CountDownLatch(slices);

        for (int s = 0; s < slices; s++) {
            final int from = s * sliceSize;
            final int to = Math.min(from + sliceSize, totalRows);
            POOL.execute(() -> {
                IN_POOL.set(Boolean.TRUE);
                try {
                    task.run(from, to);
                } finally {
                    IN_POOL.set(Boolean.FALSE);
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while waiting on parallel rows.", e);
        }
    }
}
