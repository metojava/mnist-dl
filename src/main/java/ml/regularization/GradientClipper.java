package ml.regularization;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import java.util.List;

public class GradientClipper {
    public enum Mode { NORM, VALUE }

    private final Mode mode;
    private final double threshold;

    public GradientClipper(Mode mode, double threshold) {
        this.mode = mode;
        this.threshold = threshold;
    }

    public GradientClipper(double maxNorm) {
        this(Mode.NORM, maxNorm);
    }

    public void clip(List<Layer> layers) {
        if (mode == Mode.NORM) {
            clipByNorm(layers);
        } else {
            clipByValue(layers);
        }
    }

    private void clipByNorm(List<Layer> layers) {
        double totalNorm = 0;
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense && dense.weightsGradient() != null) {
                for (int i = 0; i < dense.weightsGradient().size(); i++)
                    totalNorm += dense.weightsGradient().data()[i] * dense.weightsGradient().data()[i];
                for (int i = 0; i < dense.biasGradient().size(); i++)
                    totalNorm += dense.biasGradient().data()[i] * dense.biasGradient().data()[i];
            }
        }
        totalNorm = Math.sqrt(totalNorm);
        if (totalNorm > threshold) {
            double scale = threshold / totalNorm;
            for (Layer layer : layers) {
                if (layer instanceof DenseLayer dense && dense.weightsGradient() != null) {
                    for (int i = 0; i < dense.weightsGradient().size(); i++)
                        dense.weightsGradient().data()[i] *= scale;
                    for (int i = 0; i < dense.biasGradient().size(); i++)
                        dense.biasGradient().data()[i] *= scale;
                }
            }
        }
    }

    private void clipByValue(List<Layer> layers) {
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense && dense.weightsGradient() != null) {
                for (int i = 0; i < dense.weightsGradient().size(); i++) {
                    double v = dense.weightsGradient().data()[i];
                    dense.weightsGradient().data()[i] = Math.max(-threshold, Math.min(threshold, v));
                }
                for (int i = 0; i < dense.biasGradient().size(); i++) {
                    double v = dense.biasGradient().data()[i];
                    dense.biasGradient().data()[i] = Math.max(-threshold, Math.min(threshold, v));
                }
            }
        }
    }
}
