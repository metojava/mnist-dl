package ml.evaluation;

import ml.core.Tensor;
import ml.network.NeuralNetwork;

public class Evaluator {
    private final String[] classNames;

    public Evaluator(String[] classNames) {
        this.classNames = classNames;
    }

    public Evaluator() {
        this(null);
    }

    // ==================== EVALUATION ====================

    /**
     * Evaluates the network on the full dataset.
     *
     * The batches are processed sequentially on the calling thread: the
     * tensor kernels already parallelize internally, and submitting outer
     * batch tasks to the compute pool while forward() submits its own
     * nested tasks to the same bounded pool caused a thread-pool deadlock.
     */
    public EvaluationResult evaluate(NeuralNetwork network, Tensor images, int[] labels) {
        int numClasses = classNames != null ? classNames.length : detectClasses(labels);
        int[] predictions = network.predictBatch(images);
        ConfusionMatrix cm = new ConfusionMatrix(numClasses);
        cm.addAll(labels, predictions);
        return new EvaluationResult(labels.length, cm.correct(), cm);
    }

    // ==================== PREDICTION ====================

    public PredictionResult predict(NeuralNetwork network, Tensor images, int[] labels) {
        Tensor output = network.forward(images);
        Prediction[] preds = new Prediction[images.rows()];
        for (int i = 0; i < preds.length; i++) {
            int pred = output.argmaxRow(i);
            preds[i] = new Prediction(pred, labels[i], output.get(i, pred));
        }
        return new PredictionResult(preds);
    }

    private int detectClasses(int[] labels) {
        int max = 0;
        for (int l : labels) if (l > max) max = l;
        return max + 1;
    }
}
