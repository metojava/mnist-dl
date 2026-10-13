package ml;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * A simple 2D tensor backed by a contiguous float[] array.
 *
 * Data is stored in row-major order:
 *
 *     index = row * cols + column
 *
 * float is used instead of double because it halves memory traffic
 * and doubles the SIMD lanes per operation, which is the main win
 * for the dense matrix multiplications in this library.
 */
public class Tensor {

    private final int rows;
    private final int cols;
    private final float[] data;

    private static final Random RANDOM =
            new Random();

    // ============================================================
    // Constructors
    // ============================================================

    /**
     * Creates a zero-filled tensor.
     */
    public Tensor(
            int rows,
            int cols
    ) {

        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException(
                    "Rows and columns must be >= 0."
            );
        }

        this.rows = rows;
        this.cols = cols;

        this.data =
                new float[rows * cols];
    }

    /**
     * Creates a tensor from an existing flat array.
     *
     * The array is copied.
     */
    public Tensor(
            int rows,
            int cols,
            float[] data
    ) {

        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException(
                    "Rows and columns must be >= 0."
            );
        }

        if (data == null) {
            throw new IllegalArgumentException(
                    "Data cannot be null."
            );
        }

        if (data.length != rows * cols) {
            throw new IllegalArgumentException(
                    "Data length " +
                    data.length +
                    " does not match tensor size " +
                    (rows * cols)
            );
        }

        this.rows = rows;
        this.cols = cols;

        this.data =
                Arrays.copyOf(
                        data,
                        data.length
                );
    }

    /**
     * Creates a tensor from a 2D array.
     */
    public Tensor(
            float[][] source
    ) {

        if (source == null) {
            throw new IllegalArgumentException(
                    "Source cannot be null."
            );
        }

        this.rows =
                source.length;

        this.cols =
                rows == 0
                        ? 0
                        : source[0].length;

        this.data =
                new float[rows * cols];

        for (int i = 0;
             i < rows;
             i++) {

            if (source[i].length != cols) {
                throw new IllegalArgumentException(
                        "All rows must have the same length."
                );
            }

            System.arraycopy(
                    source[i],
                    0,
                    data,
                    i * cols,
                    cols
            );
        }
    }

    // ============================================================
    // Basic information
    // ============================================================

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public int size() {
        return data.length;
    }

    /**
     * Returns the underlying data array.
     *
     * This is intentionally not copied because neural-network
     * operations need fast access to the underlying data.
     */
    public float[] data() {
        return data;
    }

    // ============================================================
    // Element access
    // ============================================================

    /**
     * Gets one element.
     */
    public float get(
            int row,
            int col
    ) {

        return data[
                row * cols + col
        ];
    }

    /**
     * Sets one element.
     */
    public void set(
            int row,
            int col,
            float value
    ) {

        data[
                row * cols + col
        ] = value;
    }

    /**
     * Gets an element using the flat index.
     */
    public float get(
            int index
    ) {

        return data[index];
    }

    /**
     * Sets an element using the flat index.
     */
    public void set(
            int index,
            float value
    ) {

        data[index] = value;
    }

    // ============================================================
    // Factory methods
    // ============================================================

    public static Tensor zeros(
            int rows,
            int cols
    ) {

        return new Tensor(
                rows,
                cols
        );
    }

    public static Tensor ones(
            int rows,
            int cols
    ) {

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        Arrays.fill(
                result.data,
                1.0f
        );

        return result;
    }

    /**
     * Random values in [0, 1).
     */
    public static Tensor random(
            int rows,
            int cols
    ) {

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        for (int i = 0;
             i < result.data.length;
             i++) {

            result.data[i] =
                    RANDOM.nextFloat();
        }

        return result;
    }

    /**
     * Random values with Gaussian distribution.
     */
    public static Tensor randomNormal(
            int rows,
            int cols
    ) {

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        for (int i = 0;
             i < result.data.length;
             i++) {

            result.data[i] =
                    (float) RANDOM.nextGaussian();
        }

        return result;
    }

    /**
     * Xavier/Glorot initialization.
     */
    public static Tensor xavier(
            int rows,
            int cols
    ) {

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        double std =
                Math.sqrt(
                        2.0 /
                        (rows + cols)
                );

        for (int i = 0;
             i < result.data.length;
             i++) {

            result.data[i] =
                    (float) (RANDOM.nextGaussian()
                            * std);
        }

        return result;
    }

    /**
     * He initialization.
     *
     * Useful for ReLU networks.
     */
    public static Tensor he(
            int rows,
            int cols
    ) {

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        double std =
                Math.sqrt(
                        2.0 / cols
                );

        for (int i = 0;
             i < result.data.length;
             i++) {

            result.data[i] =
                    (float) (RANDOM.nextGaussian()
                            * std);
        }

        return result;
    }

    // ============================================================
    // Copy
    // ============================================================

    public Tensor copy() {

        return new Tensor(
                rows,
                cols,
                data
        );
    }

    // ============================================================
    // Matrix multiplication
    // ============================================================

    /**
     * Matrix multiplication:
     *
     *     A × B
     *
     * If A is:
     *
     *     m × n
     *
     * and B is:
     *
     *     n × p
     *
     * the result is:
     *
     *     m × p
     */
    public static Tensor matmul(
            Tensor a,
            Tensor b
    ) {

        if (a.cols != b.rows) {
            throw new IllegalArgumentException(
                    "Matrix dimensions do not match: " +
                    a.rows + "x" + a.cols +
                    " cannot be multiplied by " +
                    b.rows + "x" + b.cols
            );
        }

        Tensor result =
                new Tensor(
                        a.rows,
                        b.cols
                );

        if (result.data.length == 0) {
            return result;
        }

        matmulInto(
                a.data,
                b.data,
                result.data,
                a.rows,
                a.cols,
                b.cols
        );

        return result;
    }

    /**
     * Matrix multiplication that writes into a caller-provided
     * destination tensor, avoiding an allocation per call.
     *
     * The destination must already have the shape
     * (a.rows x b.cols); its contents are overwritten.
     */
    public static Tensor matmul(
            Tensor a,
            Tensor b,
            Tensor result
    ) {

        if (a.cols != b.rows) {
            throw new IllegalArgumentException(
                    "Matrix dimensions do not match: " +
                    a.rows + "x" + a.cols +
                    " cannot be multiplied by " +
                    b.rows + "x" + b.cols
            );
        }

        if (result.rows != a.rows ||
            result.cols != b.cols) {

            throw new IllegalArgumentException(
                    "Destination shape mismatch: expected " +
                    a.rows + "x" + b.cols +
                    " but got " +
                    result.rows + "x" + result.cols
            );
        }

        if (result.data.length == 0) {
            return result;
        }

        Arrays.fill(
                result.data,
                0.0f
        );

        matmulInto(
                a.data,
                b.data,
                result.data,
                a.rows,
                a.cols,
                b.cols
        );

        return result;
    }

    /*
     * Cache-blocked GEMM kernel.
     *
     * B is split into (j, k) tiles that stay in L1/L2 while the
     * whole batch sweeps over them, and each partial inner product
     * is accumulated in registers before being written to the
     * result once. This avoids re-reading B from main memory for
     * every row and lets the JIT vectorize the inner loop.
     */
    private static void matmulInto(
            float[] ad,
            float[] bd,
            float[] rd,
            int m,
            int n,
            int p
    ) {

        final int JC = 32;
        final int KC = 128;

        /*
         * Large kernels split the result rows across the thread
         * pool. Each thread owns its rows for every (jj, kk) tile,
         * so no two threads ever touch the same output element and
         * the per-element accumulation order is unchanged.
         */
        boolean parallel =
                m >= 64 &&
                (long) m * n * p >= 4_000_000L;

        if (parallel) {

            Parallel.parallelRows(
                    m,
                    (r0, r1) -> {

                        for (int jj = 0;
                             jj < p;
                             jj += JC) {

                            int jEnd =
                                    jj + JC < p
                                            ? jj + JC
                                            : p;

                            int jLength =
                                    jEnd - jj;

                            for (int kk = 0;
                                 kk < n;
                                 kk += KC) {

                                int kEnd =
                                        kk + KC < n
                                                ? kk + KC
                                                : n;

                                for (int i = r0;
                                     i < r1;
                                     i++) {

                                    int aRow =
                                            i * n;

                                    int resultRow =
                                            i * p + jj;

                                    int j = 0;

                                    /*
                                     * Accumulate 4 result columns
                                     * in registers so the inner
                                     * loop reads B sequentially
                                     * and touches the result array
                                     * only once.
                                     */
                                    for (;
                                         j + 4 <= jLength;
                                         j += 4) {

                                        float c0 = 0.0f;
                                        float c1 = 0.0f;
                                        float c2 = 0.0f;
                                        float c3 = 0.0f;

                                        for (int k = kk;
                                             k < kEnd;
                                             k++) {

                                            float aValue =
                                                    ad[aRow + k];

                                            int bRow =
                                                    k * p + jj + j;

                                            c0 += aValue * bd[bRow];
                                            c1 += aValue * bd[bRow + 1];
                                            c2 += aValue * bd[bRow + 2];
                                            c3 += aValue * bd[bRow + 3];
                                        }

                                        rd[resultRow + j] += c0;
                                        rd[resultRow + j + 1] += c1;
                                        rd[resultRow + j + 2] += c2;
                                        rd[resultRow + j + 3] += c3;
                                    }

                                    for (;
                                         j < jLength;
                                         j++) {

                                        float c = 0.0f;

                                        int column =
                                                jj + j;

                                        for (int k = kk;
                                             k < kEnd;
                                             k++) {

                                            c +=
                                                    ad[aRow + k]
                                                    * bd[k * p + column];
                                        }

                                        rd[resultRow + j] += c;
                                    }
                                }
                            }
                        }
                    }
            );

            return;
        }

        for (int jj = 0;
             jj < p;
             jj += JC) {

            int jEnd =
                    jj + JC < p
                            ? jj + JC
                            : p;

            int jLength =
                    jEnd - jj;

            for (int kk = 0;
                 kk < n;
                 kk += KC) {

                int kEnd =
                        kk + KC < n
                                ? kk + KC
                                : n;

                for (int i = 0;
                     i < m;
                     i++) {

                    int aRow =
                            i * n;

                    int resultRow =
                            i * p + jj;

                    int j = 0;

                    /*
                     * Accumulate 4 result columns in registers
                     * so the inner loop reads B sequentially and
                     * touches the result array only once.
                     */
                    for (;
                         j + 4 <= jLength;
                         j += 4) {

                        float c0 = 0.0f;
                        float c1 = 0.0f;
                        float c2 = 0.0f;
                        float c3 = 0.0f;

                        for (int k = kk;
                             k < kEnd;
                             k++) {

                            float aValue =
                                    ad[aRow + k];

                            int bRow =
                                    k * p + jj + j;

                            c0 += aValue * bd[bRow];
                            c1 += aValue * bd[bRow + 1];
                            c2 += aValue * bd[bRow + 2];
                            c3 += aValue * bd[bRow + 3];
                        }

                        rd[resultRow + j] += c0;
                        rd[resultRow + j + 1] += c1;
                        rd[resultRow + j + 2] += c2;
                        rd[resultRow + j + 3] += c3;
                    }

                    for (;
                         j < jLength;
                         j++) {

                        float c = 0.0f;

                        int column =
                                jj + j;

                        for (int k = kk;
                             k < kEnd;
                             k++) {

                            c +=
                                    ad[aRow + k]
                                    * bd[k * p + column];
                        }

                        rd[resultRow + j] += c;
                    }
                }
            }
        }
    }

    /**
     * Matrix multiplication with the left operand transposed:
     *
     *     A^T × B
     *
     * If A is:
     *
     *     m × n
     *
     * and B is:
     *
     *     m × p
     *
     * the result is:
     *
     *     n × p
     *
     * Used to compute dW = X^T × dZ during backpropagation without
     * materializing a copy of X^T. The batch dimension is traversed
     * in tiles so the partial results stay in cache.
     */
    public static Tensor matmulTransposedA(
            Tensor a,
            Tensor b
    ) {

        if (a.rows != b.rows) {
            throw new IllegalArgumentException(
                    "Matrix dimensions do not match: " +
                    a.rows + "x" + a.cols +
                    " cannot be multiplied by " +
                    b.rows + "x" + b.cols +
                    " with A transposed"
            );
        }

        Tensor result =
                new Tensor(
                        a.cols,
                        b.cols
                );

        if (result.data.length == 0) {
            return result;
        }

        matmulTransposedAInto(
                a.data,
                b.data,
                result.data,
                a.rows,
                a.cols,
                b.cols
        );

        return result;
    }

    /**
     * A^T × B that writes into a caller-provided destination,
     * avoiding an allocation per call.
     *
     * The destination must already have the shape
     * (a.cols x b.cols); its contents are overwritten.
     */
    public static Tensor matmulTransposedA(
            Tensor a,
            Tensor b,
            Tensor result
    ) {

        if (a.rows != b.rows) {
            throw new IllegalArgumentException(
                    "Matrix dimensions do not match: " +
                    a.rows + "x" + a.cols +
                    " cannot be multiplied by " +
                    b.rows + "x" + b.cols +
                    " with A transposed"
            );
        }

        if (result.rows != a.cols ||
            result.cols != b.cols) {

            throw new IllegalArgumentException(
                    "Destination shape mismatch: expected " +
                    a.cols + "x" + b.cols +
                    " but got " +
                    result.rows + "x" + result.cols
            );
        }

        if (result.data.length == 0) {
            return result;
        }

        Arrays.fill(
                result.data,
                0.0f
        );

        matmulTransposedAInto(
                a.data,
                b.data,
                result.data,
                a.rows,
                a.cols,
                b.cols
        );

        return result;
    }

    /*
     * Kernel for A^T × B.
     *
     * A is read column-wise (m x n stored row-major), so the batch
     * dimension is traversed in tiles of KB to keep A's column
     * slices and the partial results resident in cache.
     */
    private static void matmulTransposedAInto(
            float[] ad,
            float[] bd,
            float[] rd,
            int m,
            int n,
            int p
    ) {

        final int KB = 128;

        /*
         * The result is n x p, so rows are split across the thread
         * pool over the output rows. Each thread accumulates its
         * rows over every batch tile with no shared output element.
         */
        boolean parallel =
                n >= 64 &&
                (long) m * n * p >= 4_000_000L;

        if (parallel) {

            Parallel.parallelRows(
                    n,
                    (r0, r1) -> {

                        for (int ib = 0;
                             ib < m;
                             ib += KB) {

                            int iEnd =
                                    ib + KB < m
                                            ? ib + KB
                                            : m;

                            for (int i = r0;
                                 i < r1;
                                 i++) {

                                int resultRow =
                                        i * p;

                                for (int k = ib;
                                     k < iEnd;
                                     k++) {

                                    float aValue =
                                            ad[k * n + i];

                                    int bRow =
                                            k * p;

                                    for (int j = 0;
                                         j < p;
                                         j++) {

                                        rd[resultRow + j] +=
                                                aValue *
                                                bd[bRow + j];
                                    }
                                }
                            }
                        }
                    }
            );

            return;
        }

        for (int ib = 0;
             ib < m;
             ib += KB) {

            int iEnd =
                    ib + KB < m
                            ? ib + KB
                            : m;

            for (int i = 0;
                 i < n;
                 i++) {

                int resultRow =
                        i * p;

                for (int k = ib;
                     k < iEnd;
                     k++) {

                    float aValue =
                            ad[k * n + i];

                    int bRow =
                            k * p;

                    for (int j = 0;
                         j < p;
                         j++) {

                        rd[resultRow + j] +=
                                aValue *
                                bd[bRow + j];
                    }
                }
            }
        }
    }

    // ============================================================
    // Transpose
    // ============================================================

    public Tensor transpose() {

        Tensor result =
                new Tensor(
                        cols,
                        rows
                );

        for (int i = 0;
             i < rows;
             i++) {

            int sourceRow =
                    i * cols;

            for (int j = 0;
                 j < cols;
                 j++) {

                result.data[
                        j * rows + i
                ] =
                        data[
                                sourceRow + j
                        ];
            }
        }

        return result;
    }

    // ============================================================
    // Addition
    // ============================================================

    public Tensor add(
            Tensor other
    ) {

        checkSameShape(other);

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        for (int i = 0;
             i < data.length;
             i++) {

            result.data[i] =
                    data[i]
                    + other.data[i];
        }

        return result;
    }

    /**
     * In-place addition.
     */
    public void addInPlace(
            Tensor other
    ) {

        checkSameShape(other);

        for (int i = 0;
             i < data.length;
             i++) {

            data[i] +=
                    other.data[i];
        }
    }

    // ============================================================
    // Subtraction
    // ============================================================

    public Tensor subtract(
            Tensor other
    ) {

        checkSameShape(other);

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        for (int i = 0;
             i < data.length;
             i++) {

            result.data[i] =
                    data[i]
                    - other.data[i];
        }

        return result;
    }

    /**
     * In-place subtraction.
     */
    public void subtractInPlace(
            Tensor other
    ) {

        checkSameShape(other);

        for (int i = 0;
             i < data.length;
             i++) {

            data[i] -=
                    other.data[i];
        }
    }

    // ============================================================
    // Scalar multiplication
    // ============================================================

    public Tensor multiply(
            double scalar
    ) {

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        for (int i = 0;
             i < data.length;
             i++) {

            result.data[i] =
                    data[i] * (float) scalar;
        }

        return result;
    }

    /**
     * In-place scalar multiplication.
     */
    public void multiplyInPlace(
            double scalar
    ) {

        for (int i = 0;
             i < data.length;
             i++) {

            data[i] *= (float) scalar;
        }
    }

    // ============================================================
    // Element-wise multiplication
    // ============================================================

    public Tensor multiply(
            Tensor other
    ) {

        checkSameShape(other);

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        for (int i = 0;
             i < data.length;
             i++) {

            result.data[i] =
                    data[i]
                    * other.data[i];
        }

        return result;
    }

    // ============================================================
    // Element-wise division
    // ============================================================

    public Tensor divide(
            Tensor other
    ) {

        checkSameShape(other);

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        for (int i = 0;
             i < data.length;
             i++) {

            result.data[i] =
                    data[i]
                    / other.data[i];
        }

        return result;
    }

    // ============================================================
    // Sum
    // ============================================================

    public float sum() {

        float total = 0.0f;

        for (float value : data) {
            total += value;
        }

        return total;
    }

    // ============================================================
    // Mean
    // ============================================================

    public float mean() {

        if (data.length == 0) {
            return 0.0f;
        }

        return sum() / data.length;
    }

    // ============================================================
    // Maximum
    // ============================================================

    public float max() {

        if (data.length == 0) {
            throw new IllegalStateException(
                    "Cannot find maximum of empty tensor."
            );
        }

        float maximum =
                data[0];

        for (int i = 1;
             i < data.length;
             i++) {

            if (data[i] > maximum) {
                maximum =
                        data[i];
            }
        }

        return maximum;
    }

    // ============================================================
    // Minimum
    // ============================================================

    public float min() {

        if (data.length == 0) {
            throw new IllegalStateException(
                    "Cannot find minimum of empty tensor."
            );
        }

        float minimum =
                data[0];

        for (int i = 1;
             i < data.length;
             i++) {

            if (data[i] < minimum) {
                minimum =
                        data[i];
            }
        }

        return minimum;
    }

    // ============================================================
    // Argmax
    // ============================================================

    /**
     * Returns the column index containing the largest value
     * for a particular row.
     */
    public int argmaxRow(
            int row
    ) {

        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException(
                    "Invalid row: " + row
            );
        }

        int offset =
                row * cols;

        int bestIndex =
                0;

        float bestValue =
                data[offset];

        for (int j = 1;
             j < cols;
             j++) {

            float value =
                    data[offset + j];

            if (value > bestValue) {

                bestValue =
                        value;

                bestIndex =
                        j;
            }
        }

        return bestIndex;
    }

    // ============================================================
    // Row extraction
    // ============================================================

    /**
     * Returns a copy of one row.
     */
    public float[] row(
            int row
    ) {

        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException(
                    "Invalid row: " + row
            );
        }

        float[] result =
                new float[cols];

        System.arraycopy(
                data,
                row * cols,
                result,
                0,
                cols
        );

        return result;
    }

    // ============================================================
    // Row assignment
    // ============================================================

    public void setRow(
            int row,
            float[] values
    ) {

        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException(
                    "Invalid row: " + row
            );
        }

        if (values.length != cols) {
            throw new IllegalArgumentException(
                    "Row length does not match tensor columns."
            );
        }

        System.arraycopy(
                values,
                0,
                data,
                row * cols,
                cols
        );
    }

    // ============================================================
    // Slice rows
    // ============================================================

    /**
     * Returns rows [fromRow, toRow).
     */
    public Tensor sliceRows(
            int fromRow,
            int toRow
    ) {

        if (fromRow < 0 ||
            toRow > rows ||
            fromRow > toRow) {

            throw new IllegalArgumentException(
                    "Invalid row range."
            );
        }

        int newRows =
                toRow - fromRow;

        Tensor result =
                new Tensor(
                        newRows,
                        cols
                );

        System.arraycopy(
                data,
                fromRow * cols,
                result.data,
                0,
                newRows * cols
        );

        return result;
    }

    // ============================================================
    // Apply function
    // ============================================================

    public Tensor apply(
            DoubleUnaryOperator function
    ) {

        Tensor result =
                new Tensor(
                        rows,
                        cols
                );

        for (int i = 0;
             i < data.length;
             i++) {

            result.data[i] =
                    (float) function.applyAsDouble(
                            (double) data[i]
                    );
        }

        return result;
    }

    /**
     * In-place function application.
     */
    public void applyInPlace(
            DoubleUnaryOperator function
    ) {

        for (int i = 0;
             i < data.length;
             i++) {

            data[i] =
                    (float) function.applyAsDouble(
                            (double) data[i]
                    );
        }
    }

    // ============================================================
    // Fill
    // ============================================================

    public void fill(
            float value
    ) {

        Arrays.fill(
                data,
                value
        );
    }

    // ============================================================
    // Shape checking
    // ============================================================

    private void checkSameShape(
            Tensor other
    ) {

        if (other == null) {
            throw new IllegalArgumentException(
                    "Tensor cannot be null."
            );
        }

        if (rows != other.rows ||
            cols != other.cols) {

            throw new IllegalArgumentException(
                    "Tensor shapes do not match: " +
                    rows + "x" + cols +
                    " vs " +
                    other.rows + "x" + other.cols
            );
        }
    }

    // ============================================================
    // Conversion
    // ============================================================

    /**
     * Returns a deep 2D copy.
     *
     * Useful for visualization/debugging.
     */
    public float[][] toArray() {

        float[][] result =
                new float[rows][cols];

        for (int i = 0;
             i < rows;
             i++) {

            System.arraycopy(
                    data,
                    i * cols,
                    result[i],
                    0,
                    cols
            );
        }

        return result;
    }

    // ============================================================
    // String representation
    // ============================================================

    @Override
    public String toString() {

        StringBuilder sb =
                new StringBuilder();

        sb.append("Tensor(")
                .append(rows)
                .append("x")
                .append(cols)
                .append(")\n");

        for (int i = 0;
             i < rows;
             i++) {

            sb.append("[");

            for (int j = 0;
                 j < cols;
                 j++) {

                sb.append(
                        String.format(
                                "%.4f",
                                (double) data[i * cols + j]
                        )
                );

                if (j < cols - 1) {
                    sb.append(", ");
                }
            }

            sb.append("]");

            if (i < rows - 1) {
                sb.append("\n");
            }
        }

        return sb.toString();
    }
}