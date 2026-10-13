package ml;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Minimal row-parallel helper used by the GEMM kernels.
 *
 * Each kernel partitions the result rows across a small fixed
 * thread pool, so every thread writes disjoint output elements
 * and no synchronization is needed between accumulations. The
 * per-element summation order is unchanged, so results are
 * numerically identical to the single-threaded kernels.
 */
public final class Parallel {

    @FunctionalInterface
    public interface RangeTask {
        void run(int from, int to);
    }

    private static final int THREADS =
            Math.max(
                    1,
                    Math.min(
                            8,
                            Runtime.getRuntime()
                                    .availableProcessors()
                    )
            );

    /*
     * Daemon threads so the JVM still exits when main() returns.
     */
    private static final ExecutorService POOL =
            THREADS <= 1
                    ? null
                    : Executors.newFixedThreadPool(
                            THREADS,
                            runnable -> {
                                Thread thread =
                                        new Thread(
                                                runnable,
                                                "ml-parallel"
                                        );
                                thread.setDaemon(true);
                                return thread;
                            }
                    );

    private Parallel() {
    }

    public static int threads() {
        return THREADS;
    }

    /**
     * Runs the task on the thread pool, splitting [0, totalRows)
     * into one contiguous slice per worker thread, and blocks
     * until every slice has completed.
     */
    public static void parallelRows(
            int totalRows,
            RangeTask task
    ) {

        if (THREADS <= 1 ||
            totalRows <= 1) {

            task.run(
                    0,
                    totalRows
            );

            return;
        }

        int slices =
                Math.min(
                        THREADS,
                        totalRows
                );

        int sliceSize =
                (totalRows + slices - 1) / slices;

        CountDownLatch latch =
                new CountDownLatch(slices);

        for (int s = 0;
             s < slices;
             s++) {

            final int from =
                    s * sliceSize;

            final int to =
                    Math.min(
                            from + sliceSize,
                            totalRows
                    );

            POOL.execute(() -> {
                try {
                    task.run(
                            from,
                            to
                    );
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(
                    "Interrupted while waiting on parallel rows.",
                    e
            );
        }
    }
}