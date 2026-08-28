package ml.core;

import ml.exception.ShapeException;

import java.util.Arrays;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.DoubleUnaryOperator;

public class Tensor {
    private final int rows;
    private final int cols;
    private final double[] data;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private static final int PARALLEL_THRESHOLD = 50000;
    private static final int MATMUL_PARALLEL_OPS = 2_000_000;

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
    }

    public Tensor(double[] flat) {
        this.rows = 1;
        this.cols = flat.length;
        this.data = new double[flat.length];
        System.arraycopy(flat, 0, this.data, 0, flat.length);
    }

    public Tensor(double[][] array) {
        this.rows = array.length;
        this.cols = array[0].length;
        this.data = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(array[i], 0, data, i * cols, cols);
    }

    private Tensor(int rows, int cols, double[] data) {
        this.rows = rows;
        this.cols = cols;
        this.data = data;
    }

    // ---- ReadWriteLock for concurrent mutations ----
    public ReentrantReadWriteLock lock() { return lock; }
    public void readLock()    { lock.readLock().lock(); }
    public void readUnlock()  { lock.readLock().unlock(); }
    public void writeLock()   { lock.writeLock().lock(); }
    public void writeUnlock() { lock.writeLock().unlock(); }

    // ==================== FACTORY (parallel fill via ExecutorService) ====================

    public static Tensor zeros(int rows, int cols) { return new Tensor(rows, cols); }

    public static Tensor ones(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        Arrays.fill(t.data, 1.0);
        return t;
    }

    public static Tensor filled(int rows, int cols, double value) {
        Tensor t = new Tensor(rows, cols);
        if (t.data.length <= PARALLEL_THRESHOLD) {
            Arrays.fill(t.data, value);
        } else {
            fillParallel(t.data, value);
        }
        return t;
    }

    private static void fillParallel(double[] arr, double value) {
        ExecutorService pool = ThreadPoolManager.compute();
        int chunk = Math.max(1, arr.length / ThreadPoolManager.computePoolSize());
        CompletableFuture<?>[] futures = new CompletableFuture[(arr.length + chunk - 1) / chunk];
        for (int i = 0; i < futures.length; i++) {
            int from = i * chunk;
            int to = Math.min(from + chunk, arr.length);
            futures[i] = CompletableFuture.runAsync(() -> Arrays.fill(arr, from, to, value), pool);
        }
        CompletableFuture.allOf(futures).join();
    }

    public static Tensor random(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        int len = t.data.length;
        if (len <= PARALLEL_THRESHOLD) {
            var rng = ThreadLocalRandom.current();
            for (int i = 0; i < len; i++) t.data[i] = rng.nextDouble();
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, len / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(len + chunk - 1) / chunk];
            for (int i = 0; i < futures.length; i++) {
                int from = i * chunk;
                int to = Math.min(from + chunk, len);
                futures[i] = CompletableFuture.runAsync(() -> {
                    var rng = ThreadLocalRandom.current();
                    for (int j = from; j < to; j++) t.data[j] = rng.nextDouble();
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
        return t;
    }

    public static Tensor randomNormal(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        int len = t.data.length;
        if (len <= PARALLEL_THRESHOLD) {
            var rng = ThreadLocalRandom.current();
            for (int i = 0; i < len; i++) t.data[i] = rng.nextGaussian();
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, len / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(len + chunk - 1) / chunk];
            for (int i = 0; i < futures.length; i++) {
                int from = i * chunk;
                int to = Math.min(from + chunk, len);
                futures[i] = CompletableFuture.runAsync(() -> {
                    var rng = ThreadLocalRandom.current();
                    for (int j = from; j < to; j++) t.data[j] = rng.nextGaussian();
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
        return t;
    }

    public static Tensor xavier(int rows, int cols) {
        double std = Math.sqrt(2.0 / (rows + cols));
        Tensor t = randomNormal(rows, cols);
        t.multiplyInPlace(std);
        return t;
    }

    public static Tensor he(int rows, int cols) {
        double std = Math.sqrt(2.0 / cols);
        Tensor t = randomNormal(rows, cols);
        t.multiplyInPlace(std);
        return t;
    }

    @Deprecated public static Tensor random(int rows, int cols, java.util.Random rng) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = rng.nextDouble();
        return t;
    }
    @Deprecated public static Tensor randomNormal(int rows, int cols, java.util.Random rng) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = rng.nextGaussian();
        return t;
    }
    @Deprecated public static Tensor xavier(int rows, int cols, java.util.Random rng) {
        double std = Math.sqrt(2.0 / (rows + cols));
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = rng.nextGaussian() * std;
        return t;
    }
    @Deprecated public static Tensor he(int rows, int cols, java.util.Random rng) {
        double std = Math.sqrt(2.0 / cols);
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = rng.nextGaussian() * std;
        return t;
    }

    public static Tensor fromFlat(double[] flat, int rows, int cols) {
        if (flat.length != rows * cols)
            throw new ShapeException("fromFlat: length " + flat.length + " != " + (rows * cols));
        return new Tensor(rows, cols, Arrays.copyOf(flat, flat.length));
    }

    // ---- Shape ----
    public int rows() { return rows; }
    public int cols() { return cols; }
    public int[] shape() { return new int[]{rows, cols}; }
    public int size() { return rows * cols; }

    // ---- Data Access ----
    public double[] data() { return data; }
    public double get(int row, int col) { return data[row * cols + col]; }
    public void set(int row, int col, double value) { data[row * cols + col] = value; }

    public double[] row(int row) {
        double[] r = new double[cols];
        System.arraycopy(data, row * cols, r, 0, cols);
        return r;
    }

    public void setRow(int row, double[] values) {
        if (values.length != cols)
            throw new ShapeException("setRow: expected " + cols + ", got " + values.length);
        System.arraycopy(values, 0, data, row * cols, cols);
    }

    public Tensor sliceRows(int from, int to) {
        int count = to - from;
        double[] newData = new double[count * cols];
        System.arraycopy(data, from * cols, newData, 0, count * cols);
        return new Tensor(count, cols, newData);
    }

    // ==================== PARALLEL MATMUL (ExecutorService + CompletableFuture) ====================

    public static Tensor matmul(Tensor a, Tensor b) {
        if (a.cols != b.rows)
            throw new ShapeException("matmul: a.cols=" + a.cols + " != b.rows=" + b.rows);
        int m = a.rows, n = b.cols, k = a.cols;
        Tensor result = new Tensor(m, n);
        long ops = (long) m * n * k;
        if (ops <= MATMUL_PARALLEL_OPS) {
            matmulSequential(a.data, b.data, result.data, m, n, k, 0, m);
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, m / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(m + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, m);
                futures[t] = CompletableFuture.runAsync(
                    () -> matmulSequential(a.data, b.data, result.data, m, n, k, from, to), pool);
            }
            CompletableFuture.allOf(futures).join();
        }
        return result;
    }

    private static void matmulSequential(double[] a, double[] b, double[] c,
                                          int m, int n, int k, int rowFrom, int rowTo) {
        for (int i = rowFrom; i < rowTo; i++) {
            int ci = i * n;
            for (int p = 0; p < k; p++) {
                double aVal = a[i * k + p];
                int bp = p * n;
                for (int j = 0; j < n; j++) {
                    c[ci + j] += aVal * b[bp + j];
                }
            }
        }
    }

    public Tensor transpose() {
        Tensor result = new Tensor(cols, rows);
        if ((long) rows * cols <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result.data[j * rows + i] = data[i * cols + j];
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int chunk = Math.max(1, rows / ThreadPoolManager.computePoolSize());
            CompletableFuture<?>[] futures = new CompletableFuture[(rows + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, rows);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = from; i < to; i++)
                        for (int j = 0; j < cols; j++)
                            result.data[j * rows + i] = data[i * cols + j];
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
        return result;
    }

    // ==================== PARALLEL ELEMENT-WISE (ExecutorService) ====================

    public Tensor add(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        parallelElementwise(result.data, data, other.data, (a, b) -> a + b);
        return result;
    }

    public void addInPlace(Tensor other) {
        checkSameShape(other);
        parallelElementwiseInPlace(data, other.data, (a, b) -> a + b);
    }

    public Tensor subtract(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        parallelElementwise(result.data, data, other.data, (a, b) -> a - b);
        return result;
    }

    public void subtractInPlace(Tensor other) {
        checkSameShape(other);
        parallelElementwiseInPlace(data, other.data, (a, b) -> a - b);
    }

    public Tensor multiply(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        parallelElementwise(result.data, data, other.data, (a, b) -> a * b);
        return result;
    }

    public Tensor divide(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        parallelElementwise(result.data, data, other.data, (a, b) -> a / b);
        return result;
    }

    private void parallelElementwise(double[] out, double[] a, double[] b,
                                     java.util.function.DoubleBinaryOperator op) {
        int len = out.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) out[i] = op.applyAsDouble(a[i], b[i]);
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, len / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(len + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, len);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = from; i < to; i++) out[i] = op.applyAsDouble(a[i], b[i]);
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
    }

    private void parallelElementwiseInPlace(double[] target, double[] other,
                                            java.util.function.DoubleBinaryOperator op) {
        int len = target.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) target[i] = op.applyAsDouble(target[i], other[i]);
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, len / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(len + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, len);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = from; i < to; i++) target[i] = op.applyAsDouble(target[i], other[i]);
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
    }

    public Tensor multiply(double scalar) {
        Tensor result = new Tensor(rows, cols);
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) result.data[i] = data[i] * scalar;
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, len / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(len + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, len);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = from; i < to; i++) result.data[i] = data[i] * scalar;
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
        return result;
    }

    public void multiplyInPlace(double scalar) {
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) data[i] *= scalar;
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, len / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(len + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, len);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = from; i < to; i++) data[i] *= scalar;
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
    }

    public Tensor add(double scalar) {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) result.data[i] = data[i] + scalar;
        return result;
    }

    // ==================== PARALLEL REDUCTIONS (CompletableFuture + lock-free accumulators) ====================

    public double sum() {
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            double s = 0;
            for (double v : data) s += v;
            return s;
        }
        ExecutorService pool = ThreadPoolManager.compute();
        int threads = ThreadPoolManager.computePoolSize();
        int chunk = Math.max(1, len / threads);
        int numTasks = (len + chunk - 1) / chunk;
        @SuppressWarnings("unchecked")
        CompletableFuture<Double>[] futures = new CompletableFuture[numTasks];
        for (int t = 0; t < numTasks; t++) {
            int from = t * chunk;
            int to = Math.min(from + chunk, len);
            futures[t] = CompletableFuture.supplyAsync(() -> {
                double s = 0;
                for (int i = from; i < to; i++) s += data[i];
                return s;
            }, pool);
        }
        return CompletableFuture.allOf(futures).thenApply(v -> {
            double total = 0;
            for (var f : futures) total += f.join();
            return total;
        }).join();
    }

    public double mean() { return sum() / size(); }

    public double max() {
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            double m = Double.NEGATIVE_INFINITY;
            for (double v : data) if (v > m) m = v;
            return m;
        }
        ExecutorService pool = ThreadPoolManager.compute();
        int threads = ThreadPoolManager.computePoolSize();
        int chunk = Math.max(1, len / threads);
        int numTasks = (len + chunk - 1) / chunk;
        @SuppressWarnings("unchecked")
        CompletableFuture<Double>[] futures = new CompletableFuture[numTasks];
        for (int t = 0; t < numTasks; t++) {
            int from = t * chunk;
            int to = Math.min(from + chunk, len);
            futures[t] = CompletableFuture.supplyAsync(() -> {
                double m = Double.NEGATIVE_INFINITY;
                for (int i = from; i < to; i++) if (data[i] > m) m = data[i];
                return m;
            }, pool);
        }
        return CompletableFuture.allOf(futures).thenApply(v -> {
            double m = Double.NEGATIVE_INFINITY;
            for (var f : futures) m = Math.max(m, f.join());
            return m;
        }).join();
    }

    public double min() {
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            double m = Double.POSITIVE_INFINITY;
            for (double v : data) if (v < m) m = v;
            return m;
        }
        ExecutorService pool = ThreadPoolManager.compute();
        int threads = ThreadPoolManager.computePoolSize();
        int chunk = Math.max(1, len / threads);
        int numTasks = (len + chunk - 1) / chunk;
        @SuppressWarnings("unchecked")
        CompletableFuture<Double>[] futures = new CompletableFuture[numTasks];
        for (int t = 0; t < numTasks; t++) {
            int from = t * chunk;
            int to = Math.min(from + chunk, len);
            futures[t] = CompletableFuture.supplyAsync(() -> {
                double m = Double.POSITIVE_INFINITY;
                for (int i = from; i < to; i++) if (data[i] < m) m = data[i];
                return m;
            }, pool);
        }
        return CompletableFuture.allOf(futures).thenApply(v -> {
            double m = Double.POSITIVE_INFINITY;
            for (var f : futures) m = Math.min(m, f.join());
            return m;
        }).join();
    }

    public int argmaxRow(int row) {
        int best = 0;
        double bestVal = data[row * cols];
        for (int j = 1; j < cols; j++) {
            double v = data[row * cols + j];
            if (v > bestVal) { bestVal = v; best = j; }
        }
        return best;
    }

    public int[] argmaxAll() {
        int[] result = new int[rows];
        if (rows <= 64) {
            for (int i = 0; i < rows; i++) result[i] = argmaxRow(i);
        } else {
            parallelFor(0, rows, i -> result[i] = argmaxRow(i));
        }
        return result;
    }

    public Tensor sumRows() {
        Tensor result = new Tensor(rows, 1);
        if (rows <= 64) {
            for (int i = 0; i < rows; i++) {
                double s = 0;
                for (int j = 0; j < cols; j++) s += data[i * cols + j];
                result.data[i] = s;
            }
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int chunk = Math.max(1, rows / ThreadPoolManager.computePoolSize());
            CompletableFuture<?>[] futures = new CompletableFuture[(rows + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, rows);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = from; i < to; i++) {
                        double s = 0;
                        for (int j = 0; j < cols; j++) s += data[i * cols + j];
                        result.data[i] = s;
                    }
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
        return result;
    }

    public Tensor sumCols() {
        Tensor result = new Tensor(1, cols);
        double[] out = result.data;
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    out[j] += data[i * cols + j];
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int chunk = Math.max(1, rows / ThreadPoolManager.parallelism());
            int numFutures = (rows + chunk - 1) / chunk;
            double[][] localSums = new double[numFutures][cols];
            CompletableFuture<?>[] futures = new CompletableFuture[numFutures];
            for (int t = 0; t < numFutures; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, rows);
                final int taskIdx = t;
                futures[t] = CompletableFuture.runAsync(() -> {
                    double[] acc = localSums[taskIdx];
                    for (int i = from; i < to; i++)
                        for (int j = 0; j < cols; j++)
                            acc[j] += data[i * cols + j];
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
            for (int p = 0; p < numFutures; p++)
                for (int j = 0; j < cols; j++)
                    out[j] += localSums[p][j];
        }
        return result;
    }

    // ==================== PARALLEL FUNCTIONAL ====================

    public Tensor apply(DoubleUnaryOperator op) {
        Tensor result = new Tensor(rows, cols);
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) result.data[i] = op.applyAsDouble(data[i]);
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, len / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(len + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, len);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = from; i < to; i++) result.data[i] = op.applyAsDouble(data[i]);
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
        return result;
    }

    public void applyInPlace(DoubleUnaryOperator op) {
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) data[i] = op.applyAsDouble(data[i]);
        } else {
            ExecutorService pool = ThreadPoolManager.compute();
            int threads = ThreadPoolManager.computePoolSize();
            int chunk = Math.max(1, len / threads);
            CompletableFuture<?>[] futures = new CompletableFuture[(len + chunk - 1) / chunk];
            for (int t = 0; t < futures.length; t++) {
                int from = t * chunk;
                int to = Math.min(from + chunk, len);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = from; i < to; i++) data[i] = op.applyAsDouble(data[i]);
                }, pool);
            }
            CompletableFuture.allOf(futures).join();
        }
    }

    public Tensor clip(double min, double max) {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            double v = data[i];
            result.data[i] = v < min ? min : (v > max ? max : v);
        }
        return result;
    }

    // ==================== PARALLEL FOR UTILITY ====================

    public static void parallelFor(int from, int to, java.util.function.IntConsumer action) {
        ThreadPoolManager.parallelFor(from, to, action);
    }

    // ---- Copying ----
    public Tensor copy() { return new Tensor(rows, cols, Arrays.copyOf(data, data.length)); }

    public double[][] toArray() {
        double[][] arr = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(data, i * cols, arr[i], 0, cols);
        return arr;
    }

    // ---- Validation ----
    private void checkSameShape(Tensor other) {
        if (rows != other.rows || cols != other.cols)
            throw new ShapeException("Shape mismatch: [" + rows + ", " + cols + "] vs [" + other.rows + ", " + other.cols + "]");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Tensor(").append(rows).append("x").append(cols).append(")\n");
        int maxRows = Math.min(rows, 8);
        int maxCols = Math.min(cols, 8);
        for (int i = 0; i < maxRows; i++) {
            sb.append("[");
            for (int j = 0; j < maxCols; j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.6f", data[i * cols + j]));
            }
            if (cols > maxCols) sb.append(", ...");
            sb.append("]\n");
        }
        if (rows > maxRows) sb.append("...\n");
        return sb.toString();
    }
}
