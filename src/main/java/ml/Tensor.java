package ml;

import java.util.Arrays;
import java.util.Random;

/**
 * A simple 2D tensor backed by a contiguous double[] array.
 *
 * Data is stored in row-major order:
 *
 *     index = row * cols + column
 *
 * This representation is considerably faster than double[][] for
 * neural-network matrix operations because the data is contiguous
 * in memory and has less object overhead.
 */
public class Tensor {

    private final int rows;
    private final int cols;
    private final double[] data;

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
                new double[rows * cols];
    }

    /**
     * Creates a tensor from an existing flat array.
     *
     * The array is copied.
     */
    public Tensor(
            int rows,
            int cols,
            double[] data
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
            double[][] source
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
                new double[rows * cols];

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
    public double[] data() {
        return data;
    }

    // ============================================================
    // Element access
    // ============================================================

    /**
     * Gets one element.
     */
    public double get(
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
            double value
    ) {

        data[
                row * cols + col
        ] = value;
    }

    /**
     * Gets an element using the flat index.
     */
    public double get(
            int index
    ) {

        return data[index];
    }

    /**
     * Sets an element using the flat index.
     */
    public void set(
            int index,
            double value
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
                1.0
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
                    RANDOM.nextDouble();
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
                    RANDOM.nextGaussian();
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
                    RANDOM.nextGaussian()
                            * std;
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
                    RANDOM.nextGaussian()
                            * std;
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

        int m = a.rows;
        int n = a.cols;
        int p = b.cols;

        Tensor result =
                new Tensor(
                        m,
                        p
                );

        double[] ad =
                a.data;

        double[] bd =
                b.data;

        double[] rd =
                result.data;

        /*
         * Cache-friendly matrix multiplication.
         *
         * Instead of:
         *
         *   i -> j -> k
         *
         * we use:
         *
         *   i -> k -> j
         *
         * so B is accessed sequentially.
         */
        for (int i = 0;
             i < m;
             i++) {

            int aRow =
                    i * n;

            int resultRow =
                    i * p;

            for (int k = 0;
                 k < n;
                 k++) {

                double aValue =
                        ad[aRow + k];

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

        return result;
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
                    data[i] * scalar;
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

            data[i] *= scalar;
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

    public double sum() {

        double total = 0.0;

        for (double value : data) {
            total += value;
        }

        return total;
    }

    // ============================================================
    // Mean
    // ============================================================

    public double mean() {

        if (data.length == 0) {
            return 0.0;
        }

        return sum() / data.length;
    }

    // ============================================================
    // Maximum
    // ============================================================

    public double max() {

        if (data.length == 0) {
            throw new IllegalStateException(
                    "Cannot find maximum of empty tensor."
            );
        }

        double maximum =
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

    public double min() {

        if (data.length == 0) {
            throw new IllegalStateException(
                    "Cannot find minimum of empty tensor."
            );
        }

        double minimum =
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

        double bestValue =
                data[offset];

        for (int j = 1;
             j < cols;
             j++) {

            double value =
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
    public double[] row(
            int row
    ) {

        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException(
                    "Invalid row: " + row
            );
        }

        double[] result =
                new double[cols];

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
            double[] values
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
            java.util.function.DoubleUnaryOperator function
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
                    function.applyAsDouble(
                            data[i]
                    );
        }

        return result;
    }

    /**
     * In-place function application.
     */
    public void applyInPlace(
            java.util.function.DoubleUnaryOperator function
    ) {

        for (int i = 0;
             i < data.length;
             i++) {

            data[i] =
                    function.applyAsDouble(
                            data[i]
                    );
        }
    }

    // ============================================================
    // Fill
    // ============================================================

    public void fill(
            double value
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
    public double[][] toArray() {

        double[][] result =
                new double[rows][cols];

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
                                data[i * cols + j]
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
