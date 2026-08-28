package ml.training;

import ml.core.Tensor;
import ml.network.NeuralNetwork;

import java.util.*;

public class Metrics {
    public record MetricResult(double value) {
        @Override public String toString() { return String.format("%.4f", value); }
    }

    public static double accuracy(int[] predicted, int[] actual) {
        if (predicted.length != actual.length) throw new IllegalArgumentException("Array lengths must match");
        int correct = 0;
        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == actual[i]) correct++;
        }
        return (double) correct / predicted.length;
    }

    public static double precision(int[] predicted, int[] actual, int numClasses) {
        double totalPrecision = 0;
        int validClasses = 0;
        for (int c = 0; c < numClasses; c++) {
            int tp = 0, fp = 0;
            for (int i = 0; i < predicted.length; i++) {
                if (predicted[i] == c && actual[i] == c) tp++;
                else if (predicted[i] == c && actual[i] != c) fp++;
            }
            if (tp + fp > 0) {
                totalPrecision += (double) tp / (tp + fp);
                validClasses++;
            }
        }
        return validClasses > 0 ? totalPrecision / validClasses : 0;
    }

    public static double recall(int[] predicted, int[] actual, int numClasses) {
        double totalRecall = 0;
        int validClasses = 0;
        for (int c = 0; c < numClasses; c++) {
            int tp = 0, fn = 0;
            for (int i = 0; i < predicted.length; i++) {
                if (predicted[i] == c && actual[i] == c) tp++;
                else if (predicted[i] != c && actual[i] == c) fn++;
            }
            if (tp + fn > 0) {
                totalRecall += (double) tp / (tp + fn);
                validClasses++;
            }
        }
        return validClasses > 0 ? totalRecall / validClasses : 0;
    }

    public static double f1Score(int[] predicted, int[] actual, int numClasses) {
        double p = precision(predicted, actual, numClasses);
        double r = recall(predicted, actual, numClasses);
        return (p + r) > 0 ? 2 * p * r / (p + r) : 0;
    }

    public static double binaryAccuracy(int[] predicted, int[] actual) {
        return accuracy(predicted, actual);
    }

    public static Map<String, Double> classificationReport(int[] predicted, int[] actual, int numClasses) {
        Map<String, Double> report = new LinkedHashMap<>();
        report.put("accuracy", accuracy(predicted, actual));
        report.put("macro_precision", precision(predicted, actual, numClasses));
        report.put("macro_recall", recall(predicted, actual, numClasses));
        report.put("macro_f1", f1Score(predicted, actual, numClasses));

        for (int c = 0; c < numClasses; c++) {
            int tp = 0, fp = 0, fn = 0;
            for (int i = 0; i < predicted.length; i++) {
                if (predicted[i] == c && actual[i] == c) tp++;
                else if (predicted[i] == c && actual[i] != c) fp++;
                else if (predicted[i] != c && actual[i] == c) fn++;
            }
            double p = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0;
            double r = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0;
            double f1 = (p + r) > 0 ? 2 * p * r / (p + r) : 0;
            report.put("class_" + c + "_precision", p);
            report.put("class_" + c + "_recall", r);
            report.put("class_" + c + "_f1", f1);
            report.put("class_" + c + "_support", (double) (tp + fn));
        }
        return report;
    }

    public static Map<String, Double> classificationReport(NeuralNetwork network, Tensor x, int[] y, int numClasses) {
        int[] predictions = network.predictBatch(x);
        return classificationReport(predictions, y, numClasses);
    }

    public static void printClassificationReport(int[] predicted, int[] actual, int numClasses) {
        Map<String, Double> report = classificationReport(predicted, actual, numClasses);
        System.out.println("========================================");
        System.out.println("         Classification Report          ");
        System.out.println("========================================");
        System.out.printf("  Accuracy:        %.4f%n", report.get("accuracy"));
        System.out.printf("  Macro Precision: %.4f%n", report.get("macro_precision"));
        System.out.printf("  Macro Recall:    %.4f%n", report.get("macro_recall"));
        System.out.printf("  Macro F1:        %.4f%n", report.get("macro_f1"));
        System.out.println("----------------------------------------");
        for (int c = 0; c < numClasses; c++) {
            System.out.printf("  Class %d: P=%.4f  R=%.4f  F1=%.4f  Support=%d%n",
                c, report.get("class_" + c + "_precision"),
                report.get("class_" + c + "_recall"),
                report.get("class_" + c + "_f1"),
                report.get("class_" + c + "_support").intValue());
        }
        System.out.println("========================================");
    }

    public static void printClassificationReport(NeuralNetwork network, Tensor x, int[] y, int numClasses) {
        int[] predictions = network.predictBatch(x);
        printClassificationReport(predictions, y, numClasses);
    }

    public static Map<String, Double> topKAccuracy(Tensor predictions, int[] actual, int k) {
        int batchSize = actual.length;
        int classes = predictions.cols();
        int correct = 0;

        for (int i = 0; i < batchSize; i++) {
            double[] probs = new double[classes];
            for (int j = 0; j < classes; j++) {
                probs[j] = predictions.get(i, j);
            }

            Integer[] indices = new Integer[classes];
            for (int j = 0; j < classes; j++) indices[j] = j;
            Arrays.sort(indices, (a, b) -> Double.compare(probs[b], probs[a]));

            for (int j = 0; j < Math.min(k, classes); j++) {
                if (indices[j] == actual[i]) {
                    correct++;
                    break;
                }
            }
        }

        Map<String, Double> result = new LinkedHashMap<>();
        result.put("top_" + k + "_accuracy", (double) correct / batchSize);
        return result;
    }
}
