package ml.regularization;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import java.util.List;

public class WeightDecay {
    private final double l1;
    private final double l2;

    public WeightDecay(double l1, double l2) {
        this.l1 = l1;
        this.l2 = l2;
    }

    public WeightDecay(double l2) {
        this(0, l2);
    }

    public void apply(List<Layer> layers, double learningRate) {
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                for (int i = 0; i < dense.weights().size(); i++) {
                    double w = dense.weights().data()[i];
                    double penalty = 0;
                    if (l1 > 0) penalty += l1 * Math.signum(w);
                    if (l2 > 0) penalty += l2 * w;
                    dense.weights().data()[i] -= learningRate * penalty;
                }
            }
        }
    }

    public double computeLoss(List<Layer> layers) {
        double loss = 0;
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                for (int i = 0; i < dense.weights().size(); i++) {
                    double w = dense.weights().data()[i];
                    if (l1 > 0) loss += l1 * Math.abs(w);
                    if (l2 > 0) loss += 0.5 * l2 * w * w;
                }
            }
        }
        return loss;
    }

    public double getL1() { return l1; }
    public double getL2() { return l2; }
}
