package ml;

import java.util.Arrays;
import java.util.Random;

public class Tensor {

    private final int rows;
    private final int cols;
    private final double[][] data;

    public Tensor(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Tensor dimensions must be positive.");
        }

        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Tensor(double[][] data) {
        if (data == null || data.length == 0 || data[0].length == 0) {
            throw new IllegalArgumentException("Data cannot be empty.");
        }

        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols) {
                throw new IllegalArgumentException("All rows must have the same length.");
            }

            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }

    public static Tensor random(int rows, int cols, double min, double max) {
        Random random = new Random();
        Tensor tensor = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                tensor.data[i][j] =
                        min + random.nextDouble() * (max - min);
            }
        }

        return tensor;
    }

    public static Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    public static Tensor ones(int rows, int cols) {
        Tensor tensor = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            Arrays.fill(tensor.data[i], 1.0);
        }

        return tensor;
    }

    public double get(int row, int col) {
        return data[row][col];
    }

    public void set(int row, int col, double value) {
        data[row][col] = value;
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public double[][] getData() {
        return data;
    }

    public Tensor copy() {
        return new Tensor(data);
    }

    public Tensor add(Tensor other) {
        checkSameShape(other);

        Tensor result = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] =
                        data[i][j] + other.data[i][j];
            }
        }

        return result;
    }

    public Tensor subtract(Tensor other) {
        checkSameShape(other);

        Tensor result = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] =
                        data[i][j] - other.data[i][j];
            }
        }

        return result;
    }

    public Tensor multiply(double scalar) {
        Tensor result = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }

        return result;
    }

    public Tensor hadamard(Tensor other) {
        checkSameShape(other);

        Tensor result = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] =
                        data[i][j] * other.data[i][j];
            }
        }

        return result;
    }

    public Tensor matmul(Tensor other) {
        if (cols != other.rows) {
            throw new IllegalArgumentException(
                    "Cannot multiply " +
                    rows + "x" + cols +
                    " by " +
                    other.rows + "x" + other.cols
            );
        }

        Tensor result = new Tensor(rows, other.cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {

                double sum = 0.0;

                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * other.data[k][j];
                }

                result.data[i][j] = sum;
            }
        }

        return result;
    }

    public Tensor transpose() {
        Tensor result = new Tensor(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }

        return result;
    }

    public Tensor sumRows() {
        Tensor result = new Tensor(1, cols);

        for (int j = 0; j < cols; j++) {
            double sum = 0.0;

            for (int i = 0; i < rows; i++) {
                sum += data[i][j];
            }

            result.data[0][j] = sum;
        }

        return result;
    }

    public Tensor sumColumns() {
        Tensor result = new Tensor(rows, 1);

        for (int i = 0; i < rows; i++) {
            double sum = 0.0;

            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }

            result.data[i][0] = sum;
        }

        return result;
    }

    public double max() {
        double max = data[0][0];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                max = Math.max(max, data[i][j]);
            }
        }

        return max;
    }

    public int argMax() {
        int index = 0;
        double max = data[0][0];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                if (data[i][j] > max) {
                    max = data[i][j];
                    index = i * cols + j;
                }
            }
        }

        return index;
    }

    public void fill(double value) {
        for (int i = 0; i < rows; i++) {
            Arrays.fill(data[i], value);
        }
    }

    private void checkSameShape(Tensor other) {
        if (rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException(
                    "Tensor shapes do not match."
            );
        }
    }
	
	public int argmaxRow(int row) {

    int index = 0;
    double max = data[row][0];

    for (int j = 1; j < cols; j++) {

        if (data[row][j] > max) {
            max = data[row][j];
            index = j;
        }
    }

    return index;
}


    @Override
    public String toString() {
        return Arrays.deepToString(data);
    }
}
