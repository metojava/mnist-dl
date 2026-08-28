package ml.evaluation;

public class EvaluationResult {
    private final int total;
    private final int correct;
    private final ConfusionMatrix confusionMatrix;

    public EvaluationResult(int total, int correct, ConfusionMatrix confusionMatrix) {
        this.total = total;
        this.correct = correct;
        this.confusionMatrix = confusionMatrix;
    }

    public int total() { return total; }
    public int correct() { return correct; }
    public int incorrect() { return total - correct; }
    public double accuracy() { return (double) correct / total; }
    public double accuracyPercentage() { return accuracy() * 100; }
    public ConfusionMatrix confusionMatrix() { return confusionMatrix; }

    public double precision(int classIndex) { return confusionMatrix.precision(classIndex); }
    public double recall(int classIndex) { return confusionMatrix.recall(classIndex); }
    public double f1(int classIndex) { return confusionMatrix.f1(classIndex); }

    public double macroPrecision() {
        double sum = 0;
        for (int i = 0; i < confusionMatrix.numClasses(); i++) sum += precision(i);
        return sum / confusionMatrix.numClasses();
    }

    public double macroRecall() {
        double sum = 0;
        for (int i = 0; i < confusionMatrix.numClasses(); i++) sum += recall(i);
        return sum / confusionMatrix.numClasses();
    }

    public double macroF1() {
        double sum = 0;
        for (int i = 0; i < confusionMatrix.numClasses(); i++) sum += f1(i);
        return sum / confusionMatrix.numClasses();
    }

    @Override
    public String toString() {
        return String.format(
            "EvaluationResult{accuracy=%.2f%%, correct=%d/%d, macroPrecision=%.4f, macroRecall=%.4f, macroF1=%.4f}",
            accuracyPercentage(), correct, total, macroPrecision(), macroRecall(), macroF1());
    }
}
