package ml.core;

import ml.exception.ShapeException;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.DoubleUnaryOperator;

public class Tensor {
    private final int rows;
    private final int cols;
    private final float[] data;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private static final int PARALLEL_THRESHOLD = 50_000;
    private static final int MATMUL_PARALLEL_OPS = 4_000_000;

    /*
     * GEMM kernels. Prefer the SIMD implementation when the incubator Vector
     * API module is resolvable; otherwise fall back to the scalar kernels.
     * The reflective lookup keeps the Vector API an optional runtime
     * dependency: the library still runs on a plain JVM without
     * --add-modules jdk.incubator.vector.
     */
    private static final GemmKernels KERNELS = loadGemmKernels();

    private static GemmKernels loadGemmKernels() {
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            return VectorGemm.INSTANCE;
        } catch (Throwable t) {
            return ScalarGemm.INSTANCE;
        }
    }

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }

    public Tensor(double[] flat) {
        this.rows = 1;
        this.cols = flat.length;
        this.data = new float[flat.length];
        for (int i = 0; i < flat.length; i++) this.data[i] = (float) flat[i];
    }

    public Tensor(double[][] array) {
        this.rows = array.length;
        this.cols = array[0].length;
        this.data = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            float[] rowData = this.data;
            double[] source = array[i];
            int offset = i * cols;
            for (int j = 0; j < cols; j++) rowData[offset + j] = (float) source[j];
        }
    }

    private Tensor(int rows, int cols, float[] data) {
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

    // ==================== FACTORY ====================

    public static Tensor zeros(int rows, int cols) { return new Tensor(rows, cols); }

    public static Tensor ones(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        Arrays.fill(t.data, 1.0f);
        return t;
    }

    public static Tensor filled(int rows, int cols, double value) {
        Tensor t = new Tensor(rows, cols);
        Arrays.fill(t.data, (float) value);
        return t;
    }

    public static Tensor random(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        var rng = ThreadLocalRandom.current();
        for (int i = 0; i < t.data.length; i++) t.data[i] = rng.nextFloat();
        return t;
    }

    public static Tensor randomNormal(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        var rng = ThreadLocalRandom.current();
        for (int i = 0; i < t.data.length; i++) t.data[i] = (float) rng.nextGaussian();
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
        for (int i = 0; i < t.data.length; i++) t.data[i] = rng.nextFloat();
        return t;
    }
    @Deprecated public static Tensor randomNormal(int rows, int cols, java.util.Random rng) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = (float) rng.nextGaussian();
        return t;
    }
    @Deprecated public static Tensor xavier(int rows, int cols, java.util.Random rng) {
        double std = Math.sqrt(2.0 / (rows + cols));
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = (float) (rng.nextGaussian() * std);
        return t;
    }
    @Deprecated public static Tensor he(int rows, int cols, java.util.Random rng) {
        double std = Math.sqrt(2.0 / cols);
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = (float) (rng.nextGaussian() * std);
        return t;
    }

    public static Tensor fromFlat(double[] flat, int rows, int cols) {
        if (flat.length != rows * cols)
            throw new ShapeException("fromFlat: length " + flat.length + " != " + (rows * cols));
        float[] copy = new float[flat.length];
        for (int i = 0; i < flat.length; i++) copy[i] = (float) flat[i];
        return new Tensor(rows, cols, copy);
    }

    public static Tensor fromFlat(float[] flat, int rows, int cols) {
        if (flat.length != rows * cols)
            throw new ShapeException("fromFlat: length " + flat.length + " != " + (rows * cols));
        return new Tensor(rows, cols, flat);
    }

    // ---- Shape ----
    public int rows() { return rows; }
    public int cols() { return cols; }
    public int[] shape() { return new int[]{rows, cols}; }
    public int size() { return rows * cols; }

    // ---- Data Access ----
    public float[] data() { return data; }
    public double get(int row, int col) { return data[row * cols + col]; }
    public void set(int row, int col, double value) { data[row * cols + col] = (float) value; }

    public double[] row(int row) {
        double[] r = new double[cols];
        int offset = row * cols;
        for (int j = 0; j < cols; j++) r[j] = data[offset + j];
        return r;
    }

    public void setRow(int row, double[] values) {
        if (values.length != cols)
            throw new ShapeException("setRow: expected " + cols + ", got " + values.length);
        int offset = row * cols;
        for (int j = 0; j < cols; j++) data[offset + j] = (float) values[j];
    }

    public Tensor sliceRows(int from, int to) {
        int count = to - from;
        float[] newData = new float[count * cols];
        System.arraycopy(data, from * cols, newData, 0, count * cols);
        return new Tensor(count, cols, newData);
    }

    // ==================== MATMUL (cache-blocked) ====================

    public static Tensor matmul(Tensor a, Tensor b) {
        if (a.cols != b.rows)
            throw new ShapeException("matmul: a.cols=" + a.cols + " != b.rows=" + b.rows);
        Tensor result = new Tensor(a.rows, b.cols);
        matmulInto(a.data, b.data, result.data, a.rows, a.cols, b.cols);
        return result;
    }

    /**
     * Matrix multiplication writing into a caller-provided destination,
     * avoiding an allocation per call. Destination contents are overwritten.
     */
    public static Tensor matmul(Tensor a, Tensor b, Tensor result) {
        if (a.cols != b.rows)
            throw new ShapeException("matmul: a.cols=" + a.cols + " != b.rows=" + b.rows);
        if (result.rows != a.rows || result.cols != b.cols)
            throw new ShapeException("matmul: bad destination " + result.rows + "x" + result.cols
                    + ", expected " + a.rows + "x" + b.cols);
        Arrays.fill(result.data, 0.0f);
        matmulInto(a.data, b.data, result.data, a.rows, a.cols, b.cols);
        return result;
    }

    /*
     * C = A*B  (A: m x n, B: n x p, C: m x p), row-major.
     * Cache-blocked over (j, k) with 4-column register accumulation so B is
     * read sequentially and each result element is touched once per tile.
     * Row slices are distributed across the pool; per-element accumulation
     * order is unchanged, so results match the naive kernel exactly.
     */
    private static void matmulInto(float[] ad, float[] bd, float[] rd, int m, int n, int p) {
        boolean parallel = m >= 64 && (long) m * n * p >= MATMUL_PARALLEL_OPS;
        if (parallel) {
            Parallel.parallelRows(m, (r0, r1) -> KERNELS.gemmRows(ad, bd, rd, m, n, p, r0, r1));
        } else {
            KERNELS.gemmRows(ad, bd, rd, m, n, p, 0, m);
        }
    }

    // ==================== matmul with A transposed: A^T * B ====================

    /**
     * C = A^T * B  (A: m x n, B: m x p, C: n x p). Avoids materializing A^T;
     * used for dW = X^T dZ during backpropagation.
     */
    public static Tensor matmulTransposedA(Tensor a, Tensor b) {
        if (a.rows != b.rows)
            throw new ShapeException("matmulTransposedA: a.rows=" + a.rows + " != b.rows=" + b.rows);
        Tensor result = new Tensor(a.cols, b.cols);
        matmulTransposedAInto(a.data, b.data, result.data, a.rows, a.cols, b.cols);
        return result;
    }

    public static Tensor matmulTransposedA(Tensor a, Tensor b, Tensor result) {
        if (a.rows != b.rows)
            throw new ShapeException("matmulTransposedA: a.rows=" + a.rows + " != b.rows=" + b.rows);
        if (result.rows != a.cols || result.cols != b.cols)
            throw new ShapeException("matmulTransposedA: bad destination " + result.rows + "x" + result.cols
                    + ", expected " + a.cols + "x" + b.cols);
        Arrays.fill(result.data, 0.0f);
        matmulTransposedAInto(a.data, b.data, result.data, a.rows, a.cols, b.cols);
        return result;
    }

    private static void matmulTransposedAInto(float[] ad, float[] bd, float[] rd,
                                              int m, int n, int p) {
        boolean parallel = n >= 64 && (long) m * n * p >= MATMUL_PARALLEL_OPS;
        if (parallel) {
            Parallel.parallelRows(n, (r0, r1) -> KERNELS.transposedARows(ad, bd, rd, m, n, p, r0, r1));
        } else {
            KERNELS.transposedARows(ad, bd, rd, m, n, p, 0, n);
        }
    }

    // ==================== TRANSPOSE ====================

    public Tensor transpose() {
        Tensor result = new Tensor(cols, rows);
        float[] rd = result.data;
        if ((long) rows * cols <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    rd[j * rows + i] = data[i * cols + j];
        } else {
            Parallel.parallelRows(rows, (from, to) -> {
                for (int i = from; i < to; i++)
                    for (int j = 0; j < cols; j++)
                        rd[j * rows + i] = data[i * cols + j];
            });
        }
        return result;
    }

    // ==================== ELEMENT-WISE ====================

    public Tensor add(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        float[] out = result.data;
        int len = out.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) out[i] = data[i] + other.data[i];
        } else {
            Parallel.parallelRows(len, (from, to) -> {
                for (int i = from; i < to; i++) out[i] = data[i] + other.data[i];
            });
        }
        return result;
    }

    public void addInPlace(Tensor other) {
        checkSameShape(other);
        int len = data.length;
        for (int i = 0; i < len; i++) data[i] += other.data[i];
    }

    public Tensor subtract(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        float[] out = result.data;
        int len = out.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) out[i] = data[i] - other.data[i];
        } else {
            Parallel.parallelRows(len, (from, to) -> {
                for (int i = from; i < to; i++) out[i] = data[i] - other.data[i];
            });
        }
        return result;
    }

    public void subtractInPlace(Tensor other) {
        checkSameShape(other);
        int len = data.length;
        for (int i = 0; i < len; i++) data[i] -= other.data[i];
    }

    public Tensor multiply(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        float[] out = result.data;
        int len = out.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) out[i] = data[i] * other.data[i];
        } else {
            Parallel.parallelRows(len, (from, to) -> {
                for (int i = from; i < to; i++) out[i] = data[i] * other.data[i];
            });
        }
        return result;
    }

    public Tensor divide(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        float[] out = result.data;
        int len = out.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) out[i] = data[i] / other.data[i];
        } else {
            Parallel.parallelRows(len, (from, to) -> {
                for (int i = from; i < to; i++) out[i] = data[i] / other.data[i];
            });
        }
        return result;
    }

    public Tensor multiply(double scalar) {
        Tensor result = new Tensor(rows, cols);
        float s = (float) scalar;
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) result.data[i] = data[i] * s;
        } else {
            Parallel.parallelRows(len, (from, to) -> {
                for (int i = from; i < to; i++) result.data[i] = data[i] * s;
            });
        }
        return result;
    }

    public void multiplyInPlace(double scalar) {
        float s = (float) scalar;
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) data[i] *= s;
        } else {
            Parallel.parallelRows(len, (from, to) -> {
                for (int i = from; i < to; i++) data[i] *= s;
            });
        }
    }

    public Tensor add(double scalar) {
        Tensor result = new Tensor(rows, cols);
        float s = (float) scalar;
        for (int i = 0; i < data.length; i++) result.data[i] = data[i] + s;
        return result;
    }

    // ==================== REDUCTIONS ====================

    public double sum() {
        double s = 0;
        for (float v : data) s += v;
        return s;
    }

    public double mean() { return sum() / size(); }

    public double max() {
        float m = Float.NEGATIVE_INFINITY;
        for (float v : data) if (v > m) m = v;
        return m;
    }

    public double min() {
        float m = Float.POSITIVE_INFINITY;
        for (float v : data) if (v < m) m = v;
        return m;
    }

    public int argmaxRow(int row) {
        int best = 0;
        float bestVal = data[row * cols];
        for (int j = 1; j < cols; j++) {
            float v = data[row * cols + j];
            if (v > bestVal) { bestVal = v; best = j; }
        }
        return best;
    }

    public int[] argmaxAll() {
        int[] result = new int[rows];
        for (int i = 0; i < rows; i++) result[i] = argmaxRow(i);
        return result;
    }

    public Tensor sumRows() {
        Tensor result = new Tensor(rows, 1);
        float[] out = result.data;
        for (int i = 0; i < rows; i++) {
            float s = 0;
            int off = i * cols;
            for (int j = 0; j < cols; j++) s += data[off + j];
            out[i] = s;
        }
        return result;
    }

    public Tensor sumCols() {
        Tensor result = new Tensor(1, cols);
        float[] out = result.data;
        for (int i = 0; i < rows; i++) {
            int off = i * cols;
            for (int j = 0; j < cols; j++) out[j] += data[off + j];
        }
        return result;
    }

    // ==================== FUNCTIONAL ====================

    public Tensor apply(DoubleUnaryOperator op) {
        Tensor result = new Tensor(rows, cols);
        float[] out = result.data;
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) out[i] = (float) op.applyAsDouble(data[i]);
        } else {
            Parallel.parallelRows(len, (from, to) -> {
                for (int i = from; i < to; i++) out[i] = (float) op.applyAsDouble(data[i]);
            });
        }
        return result;
    }

    public void applyInPlace(DoubleUnaryOperator op) {
        int len = data.length;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = 0; i < len; i++) data[i] = (float) op.applyAsDouble(data[i]);
        } else {
            Parallel.parallelRows(len, (from, to) -> {
                for (int i = from; i < to; i++) data[i] = (float) op.applyAsDouble(data[i]);
            });
        }
    }

    public Tensor clip(double min, double max) {
        Tensor result = new Tensor(rows, cols);
        float lo = (float) min, hi = (float) max;
        for (int i = 0; i < data.length; i++) {
            float v = data[i];
            result.data[i] = v < lo ? lo : (v > hi ? hi : v);
        }
        return result;
    }

    // ==================== PARALLEL FOR UTILITY ====================

    public static void parallelFor(int from, int to, java.util.function.IntConsumer action) {
        int len = to - from;
        if (len <= 0) return;
        if (len <= PARALLEL_THRESHOLD) {
            for (int i = from; i < to; i++) action.accept(i);
        } else {
            Parallel.parallelRows(len, (a, b) -> {
                for (int i = a; i < b; i++) action.accept(from + i);
            });
        }
    }

    // ---- Copying ----
    public Tensor copy() { return new Tensor(rows, cols, Arrays.copyOf(data, data.length)); }

    public double[][] toArray() {
        double[][] arr = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            int off = i * cols;
            for (int j = 0; j < cols; j++) arr[i][j] = data[off + j];
        }
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
                sb.append(String.format("%.6f", (double) data[i * cols + j]));
            }
            if (cols > maxCols) sb.append(", ...");
            sb.append("]\n");
        }
        if (rows > maxRows) sb.append("...\n");
        return sb.toString();
    }
}
