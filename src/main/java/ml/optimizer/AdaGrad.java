package ml.optimizer;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdaGrad implements Optimizer {
    private double learningRate;
    private final double epsilon;
    private List<double[]> cacheW, cacheB;
    private boolean initialized;

    public AdaGrad(double learningRate, double epsilon) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
        this.initialized = false;
    }

    public AdaGrad(double learningRate) {
        this(learningRate, 1e-8);
    }

    @Override
    public void step(List<Layer> layers, double lr) {
        double effectiveLr = lr > 0 ? lr : this.learningRate;
        if (!initialized) init(layers);
        int idx = 0;
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                if (dense.weightsGradient() == null) { idx++; continue; }
                for (int i = 0; i < dense.weights().size(); i++) {
                    double g = dense.weightsGradient().data()[i];
                    cacheW.get(idx)[i] += g * g;
                    dense.weights().data()[i] -= effectiveLr * g / (Math.sqrt(cacheW.get(idx)[i]) + epsilon);
                }
                for (int i = 0; i < dense.bias().size(); i++) {
                    double g = dense.biasGradient().data()[i];
                    cacheB.get(idx)[i] += g * g;
                    dense.bias().data()[i] -= effectiveLr * g / (Math.sqrt(cacheB.get(idx)[i]) + epsilon);
                }
                dense.invalidateTranspose();
                idx++;
            }
        }
    }

    private void init(List<Layer> layers) {
        cacheW = new ArrayList<>(); cacheB = new ArrayList<>();
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                cacheW.add(new double[dense.weights().size()]);
                cacheB.add(new double[dense.bias().size()]);
            }
        }
        initialized = true;
    }

    @Override
    public void setLearningRate(double lr) { this.learningRate = lr; }
    @Override
    public double getLearningRate() { return learningRate; }

    public void saveState(DataOutputStream dos) throws IOException {
        dos.writeInt(cacheW.size());
        for (int i = 0; i < cacheW.size(); i++) {
            ml.network.ModelSerializer.writeDoubles(dos, cacheW.get(i));
            ml.network.ModelSerializer.writeDoubles(dos, cacheB.get(i));
        }
    }

    public void loadState(DataInputStream dis) throws IOException {
        int count = dis.readInt();
        cacheW = new ArrayList<>(); cacheB = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            cacheW.add(ml.network.ModelSerializer.readDoubles(dis));
            cacheB.add(ml.network.ModelSerializer.readDoubles(dis));
        }
        initialized = true;
    }
}
