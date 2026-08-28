package ml;

import java.util.Arrays;

public class ConfusionMatrix {

    private final int numClasses;
    private final int[][] matrix;

    public ConfusionMatrix(int numClasses) {
        if (numClasses <= 0) {
            throw new IllegalArgumentException(
                    "Number of classes must be greater than zero."
            );
        }

        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
    }

    /**
     * Add one prediction to the confusion matrix.
     *
     * Rows    = actual labels
     * Columns = predicted labels
     */
    public void add(
            int actual,
            int predicted
    ) {
        if (actual < 0 || actual >= numClasses) {
            throw new IllegalArgumentException(
                    "Invalid actual label: " + actual
            );
        }

        if (predicted < 0 || predicted >= numClasses) {
            throw new IllegalArgumentException(
                    "Invalid predicted label: " + predicted
            );
        }

        matrix[actual][predicted]++;
    }

    /**
     * Add an entire set of predictions.
     */
    public void addAll(
            int[] actual,
            int[] predicted
    ) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException(
                    "Actual and predicted arrays must have the same length."
            );
        }

        for (int i = 0; i < actual.length; i++) {
            add(
                    actual[i],
                    predicted[i]
            );
        }
    }

    public int get(
            int actual,
            int predicted
    ) {
        return matrix[actual][predicted];
    }

    public int numClasses() {
        return numClasses;
    }

    /**
     * Number of samples belonging to an actual class.
     */
    public int actualCount(int actual) {

        int total = 0;

        for (int predicted = 0;
             predicted < numClasses;
             predicted++) {

            total += matrix[actual][predicted];
        }

        return total;
    }

    /**
     * Number of samples predicted as a class.
     */
    public int predictedCount(int predicted) {

        int total = 0;

        for (int actual = 0;
             actual < numClasses;
             actual++) {

            total += matrix[actual][predicted];
        }

        return total;
    }

    /**
     * True positives for one class.
     */
    public int truePositive(int classIndex) {
        return matrix[classIndex][classIndex];
    }

    /**
     * False positives for one class.
     */
    public int falsePositive(int classIndex) {

        int total = predictedCount(classIndex);

        return total - truePositive(classIndex);
    }

    /**
     * False negatives for one class.
     */
    public int falseNegative(int classIndex) {

        int total = actualCount(classIndex);

        return total - truePositive(classIndex);
    }

    /**
     * Precision for one class.
     */
    public double precision(int classIndex) {

        int tp = truePositive(classIndex);
        int fp = falsePositive(classIndex);

        if (tp + fp == 0) {
            return 0.0;
        }

        return (double) tp / (tp + fp);
    }

    /**
     * Recall for one class.
     */
    public double recall(int classIndex) {

        int tp = truePositive(classIndex);
        int fn = falseNegative(classIndex);

        if (tp + fn == 0) {
            return 0.0;
        }

        return (double) tp / (tp + fn);
    }

    /**
     * F1 score for one class.
     */
    public double f1(int classIndex) {

        double precision =
                precision(classIndex);

        double recall =
                recall(classIndex);

        if (precision + recall == 0.0) {
            return 0.0;
        }

        return 2.0 *
                precision *
                recall /
                (precision + recall);
    }

    /**
     * Total number of samples.
     */
    public int total() {

        int total = 0;

        for (int actual = 0;
             actual < numClasses;
             actual++) {

            for (int predicted = 0;
                 predicted < numClasses;
                 predicted++) {

                total += matrix[actual][predicted];
            }
        }

        return total;
    }

    /**
     * Number of correct predictions.
     */
    public int correct() {

        int correct = 0;

        for (int i = 0;
             i < numClasses;
             i++) {

            correct += matrix[i][i];
        }

        return correct;
    }

    /**
     * Number of incorrect predictions.
     */
    public int incorrect() {
        return total() - correct();
    }

    /**
     * Overall accuracy.
     */
    public double accuracy() {

        int total = total();

        if (total == 0) {
            return 0.0;
        }

        return (double) correct() / total;
    }

    public double accuracyPercentage() {
        return accuracy() * 100.0;
    }

    /**
     * Returns a copy of the underlying matrix.
     */
    public int[][] toArray() {

        int[][] copy =
                new int[numClasses][numClasses];

        for (int i = 0;
             i < numClasses;
             i++) {

            copy[i] =
                    Arrays.copyOf(
                            matrix[i],
                            numClasses
                    );
        }

        return copy;
    }

    /**
     * Prints the matrix in a DL4J-like format.
     */
    public void print() {

        System.out.println();
        System.out.println(
                "================ CONFUSION MATRIX ================"
        );

        System.out.println(
                "Rows = Actual, Columns = Predicted"
        );

        System.out.println();

        System.out.printf(
                "%8s",
                ""
        );

        for (int predicted = 0;
             predicted < numClasses;
             predicted++) {

            System.out.printf(
                    "%8d",
                    predicted
            );
        }

        System.out.println();

        for (int actual = 0;
             actual < numClasses;
             actual++) {

            System.out.printf(
                    "%8d",
                    actual
            );

            for (int predicted = 0;
                 predicted < numClasses;
                 predicted++) {

                System.out.printf(
                        "%8d",
                        matrix[actual][predicted]
                );
            }

            System.out.println();
        }

        System.out.println();

        System.out.printf(
                "Total samples : %d%n",
                total()
        );

        System.out.printf(
                "Correct       : %d%n",
                correct()
        );

        System.out.printf(
                "Incorrect     : %d%n",
                incorrect()
        );

        System.out.printf(
                "Accuracy      : %.2f%%%n",
                accuracyPercentage()
        );

        System.out.println(
                "==================================================="
        );
    }

    /**
     * Prints per-class precision, recall and F1.
     */
    public void printClassMetrics() {

        System.out.println();
        System.out.println(
                "================ CLASS METRICS ==================="
        );

        System.out.printf(
                "%8s %12s %12s %12s%n",
                "Class",
                "Precision",
                "Recall",
                "F1"
        );

        for (int i = 0;
             i < numClasses;
             i++) {

            System.out.printf(
                    "%8d %11.2f%% %11.2f%% %11.2f%%%n",
                    i,
                    precision(i) * 100.0,
                    recall(i) * 100.0,
                    f1(i) * 100.0
            );
        }

        System.out.println(
                "==================================================="
        );
    }
}
