package ml.optimizer;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;

public class SGD implements Optimizer {
    private double learningRate;
    private double momentum;
    private boolean initialized;
    private List<double[]> velocityW;
    private List<double[]> velocityB;

    public SGD(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.initialized = false;
    }

    public SGD(double learningRate) {
        this(learningRate, 0.0);
    }

    public SGD() {
        this(0.01, 0.0);
    }

    @Override
    public void step(List<Layer> layers, double lr) {
        double effectiveLr = lr > 0 ? lr : this.learningRate;
        if (momentum > 0 && !initialized) {
            initVelocity(layers);
            initialized = true;
        }
        int idx = 0;
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                if (dense.weightsGradient() != null && momentum > 0 && velocityW != null) {
                    for (int i = 0; i < dense.weights().size(); i++) {
                        velocityW.get(idx)[i] = momentum * velocityW.get(idx)[i] + effectiveLr * dense.weightsGradient().data()[i];
                        dense.weights().data()[i] -= velocityW.get(idx)[i];
                    }
                    for (int i = 0; i < dense.bias().size(); i++) {
                        velocityB.get(idx)[i] = momentum * velocityB.get(idx)[i] + effectiveLr * dense.biasGradient().data()[i];
                        dense.bias().data()[i] -= velocityB.get(idx)[i];
                    }
                    dense.invalidateTranspose();
                } else {
                    layer.update(effectiveLr);
                }
                idx++;
            }
        }
    }

    private void initVelocity(List<Layer> layers) {
        velocityW = new java.util.ArrayList<>();
        velocityB = new java.util.ArrayList<>();
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                velocityW.add(new double[dense.weights().size()]);
                velocityB.add(new double[dense.bias().size()]);
            }
        }
    }

    @Override
    public void setLearningRate(double lr) { this.learningRate = lr; }
    @Override
    public double getLearningRate() { return learningRate; }

    public void saveState(DataOutputStream dos) throws IOException {
        dos.writeDouble(momentum);
        dos.writeBoolean(initialized);
        if (initialized && velocityW != null) {
            dos.writeInt(velocityW.size());
            for (int i = 0; i < velocityW.size(); i++) {
                ml.network.ModelSerializer.writeDoubles(dos, velocityW.get(i));
                ml.network.ModelSerializer.writeDoubles(dos, velocityB.get(i));
            }
        } else {
            dos.writeInt(0);
        }
    }

    public void loadState(DataInputStream dis) throws IOException {
        this.momentum = dis.readDouble();
        this.initialized = dis.readBoolean();
        int count = dis.readInt();
        if (count > 0) {
            velocityW = new java.util.ArrayList<>();
            velocityB = new java.util.ArrayList<>();
            for (int i = 0; i < count; i++) {
                velocityW.add(ml.network.ModelSerializer.readDoubles(dis));
                velocityB.add(ml.network.ModelSerializer.readDoubles(dis));
            }
        }
    }
}
