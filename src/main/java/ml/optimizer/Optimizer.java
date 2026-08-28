package ml.optimizer;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import java.util.List;

public interface Optimizer {
    void step(List<Layer> layers, double learningRate);
    default void zeroGradients() {}
    void setLearningRate(double lr);
    double getLearningRate();
}
