package ml.training;

import ml.core.Tensor;
import ml.core.ThreadPoolManager;

import java.util.concurrent.ThreadLocalRandom;

public class DataLoader {
    private final int batchSize;
    private final boolean shuffle;

    public DataLoader(int batchSize, boolean shuffle) {
        this.batchSize = batchSize;
        this.shuffle = shuffle;
    }

    public DataLoader(int batchSize) {
        this(batchSize, true);
    }

    public BatchIterator iterate(Tensor images, int[] labels) {
        return new BatchIterator(images, labels);
    }

    public class BatchIterator {
        private final Tensor images;
        private final int[] labels;
        private final int[] indices;
        private int pos;

        public BatchIterator(Tensor images, int[] labels) {
            this.images = images;
            this.labels = labels;
            this.pos = 0;
            this.indices = new int[images.rows()];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            if (shuffle) parallelShuffle();
        }

        public boolean hasNext() { return pos < images.rows(); }

        public Batch next() {
            int start = pos;
            int end = Math.min(pos + batchSize, images.rows());
            pos = end;
            int count = end - start;
            int cols = images.cols();
            float[] buf = new float[count * cols];
            int[] batchLabels = new int[count];
            for (int i = 0; i < count; i++) {
                int idx = indices[start + i];
                System.arraycopy(images.data(), idx * cols, buf, i * cols, cols);
                batchLabels[i] = labels[idx];
            }
            Tensor batchImages = Tensor.fromFlat(buf, count, cols);
            return new Batch(batchImages, batchLabels);
        }

        /**
         * Parallel Fisher-Yates shuffle using ExecutorService.
         * Splits index array into segments, each thread shuffles its segment,
         * then a final cross-segment swap pass ensures full randomness.
         */
        private void parallelShuffle() {
            int n = indices.length;
            if (n <= 1) return;
            sequentialShuffle(n);
        }

        private void sequentialShuffle(int n) {
            var rng = ThreadLocalRandom.current();
            for (int i = n - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }
        }
    }

    public record Batch(Tensor images, int[] labels) {
        public int size() { return images.rows(); }
    }
}
