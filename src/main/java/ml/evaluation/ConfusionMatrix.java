package ml.evaluation;

public class ConfusionMatrix {
    private final int numClasses;
    private final int[][] matrix;

    public ConfusionMatrix(int numClasses) {
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
    }

    public void add(int actual, int predicted) {
        if (actual >= 0 && actual < numClasses && predicted >= 0 && predicted < numClasses)
            matrix[actual][predicted]++;
    }

    public void addAll(int[] actual, int[] predicted) {
        for (int i = 0; i < actual.length; i++) add(actual[i], predicted[i]);
    }

    public int get(int actual, int predicted) { return matrix[actual][predicted]; }

    public int numClasses() { return numClasses; }

    public int actualCount(int classIndex) {
        int sum = 0;
        for (int j = 0; j < numClasses; j++) sum += matrix[classIndex][j];
        return sum;
    }

    public int predictedCount(int classIndex) {
        int sum = 0;
        for (int i = 0; i < numClasses; i++) sum += matrix[i][classIndex];
        return sum;
    }

    public int truePositive(int classIndex) { return matrix[classIndex][classIndex]; }

    public int falsePositive(int classIndex) {
        int sum = 0;
        for (int i = 0; i < numClasses; i++)
            if (i != classIndex) sum += matrix[i][classIndex];
        return sum;
    }

    public int falseNegative(int classIndex) {
        int sum = 0;
        for (int j = 0; j < numClasses; j++)
            if (j != classIndex) sum += matrix[classIndex][j];
        return sum;
    }

    public double precision(int classIndex) {
        int denom = truePositive(classIndex) + falsePositive(classIndex);
        return denom == 0 ? 0 : (double) truePositive(classIndex) / denom;
    }

    public double recall(int classIndex) {
        int denom = truePositive(classIndex) + falseNegative(classIndex);
        return denom == 0 ? 0 : (double) truePositive(classIndex) / denom;
    }

    public double f1(int classIndex) {
        double p = precision(classIndex);
        double r = recall(classIndex);
        return (p + r) == 0 ? 0 : 2 * p * r / (p + r);
    }

    public int total() {
        int sum = 0;
        for (int[] row : matrix) for (int v : row) sum += v;
        return sum;
    }

    public int correct() {
        int sum = 0;
        for (int i = 0; i < numClasses; i++) sum += matrix[i][i];
        return sum;
    }

    public int incorrect() { return total() - correct(); }
    public double accuracy() { return (double) correct() / total(); }

    public int[][] toArray() {
        int[][] copy = new int[numClasses][numClasses];
        for (int i = 0; i < numClasses; i++)
            System.arraycopy(matrix[i], 0, copy[i], 0, numClasses);
        return copy;
    }

    public void print() {
        System.out.println("Confusion Matrix:");
        System.out.printf("%-10s", "");
        for (int j = 0; j < numClasses; j++) System.out.printf("%8d", j);
        System.out.println();
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%-10d", i);
            for (int j = 0; j < numClasses; j++) System.out.printf("%8d", matrix[i][j]);
            System.out.println();
        }
    }

    public void printClassMetrics(String[] classNames) {
        System.out.printf("%-20s %10s %10s %10s%n", "Class", "Precision", "Recall", "F1");
        System.out.println("-".repeat(52));
        for (int i = 0; i < numClasses; i++) {
            String name = classNames != null && i < classNames.length ? classNames[i] : "Class " + i;
            System.out.printf("%-20s %10.4f %10.4f %10.4f%n", name, precision(i), recall(i), f1(i));
        }
    }
}
