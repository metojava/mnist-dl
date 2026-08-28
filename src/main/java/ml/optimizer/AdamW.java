package ml.optimizer;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdamW implements Optimizer {
    private double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private final double weightDecay;
    private int timestep;
    private List<double[]> mW, vW;
    private List<double[]> mB, vB;
    private boolean initialized;

    public AdamW(double learningRate, double beta1, double beta2, double epsilon, double weightDecay) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.timestep = 0;
        this.initialized = false;
    }

    public AdamW(double learningRate, double weightDecay) {
        this(learningRate, 0.9, 0.999, 1e-8, weightDecay);
    }

    public AdamW(double learningRate) {
        this(learningRate, 0.01);
    }

    @Override
    public void step(List<Layer> layers, double lr) {
        double effectiveLr = lr > 0 ? lr : this.learningRate;
        if (!initialized) init(layers);
        timestep++;
        double bc1 = 1.0 - Math.pow(beta1, timestep);
        double bc2 = 1.0 - Math.pow(beta2, timestep);
        int idx = 0;
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                if (dense.weightsGradient() == null) { idx++; continue; }
                for (int i = 0; i < dense.weights().size(); i++) {
                    double g = dense.weightsGradient().data()[i];
                    mW.get(idx)[i] = beta1 * mW.get(idx)[i] + (1 - beta1) * g;
                    vW.get(idx)[i] = beta2 * vW.get(idx)[i] + (1 - beta2) * g * g;
                    double mHat = mW.get(idx)[i] / bc1;
                    double vHat = vW.get(idx)[i] / bc2;
                    dense.weights().data()[i] -= effectiveLr * (mHat / (Math.sqrt(vHat) + epsilon) + weightDecay * dense.weights().data()[i]);
                }
                for (int i = 0; i < dense.bias().size(); i++) {
                    double g = dense.biasGradient().data()[i];
                    mB.get(idx)[i] = beta1 * mB.get(idx)[i] + (1 - beta1) * g;
                    vB.get(idx)[i] = beta2 * vB.get(idx)[i] + (1 - beta2) * g * g;
                    double mHat = mB.get(idx)[i] / bc1;
                    double vHat = vB.get(idx)[i] / bc2;
                    dense.bias().data()[i] -= effectiveLr * mHat / (Math.sqrt(vHat) + epsilon);
                }
                dense.invalidateTranspose();
                idx++;
            }
        }
    }

    private void init(List<Layer> layers) {
        mW = new ArrayList<>(); vW = new ArrayList<>();
        mB = new ArrayList<>(); vB = new ArrayList<>();
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                mW.add(new double[dense.weights().size()]);
                vW.add(new double[dense.weights().size()]);
                mB.add(new double[dense.bias().size()]);
                vB.add(new double[dense.bias().size()]);
            }
        }
        initialized = true;
    }

    @Override
    public void setLearningRate(double lr) { this.learningRate = lr; }
    @Override
    public double getLearningRate() { return learningRate; }

    public void saveState(DataOutputStream dos) throws IOException {
        dos.writeInt(timestep);
        dos.writeDouble(weightDecay);
        dos.writeInt(mW.size());
        for (int i = 0; i < mW.size(); i++) {
            ml.network.ModelSerializer.writeDoubles(dos, mW.get(i));
            ml.network.ModelSerializer.writeDoubles(dos, vW.get(i));
            ml.network.ModelSerializer.writeDoubles(dos, mB.get(i));
            ml.network.ModelSerializer.writeDoubles(dos, vB.get(i));
        }
    }

    public void loadState(DataInputStream dis) throws IOException {
        timestep = dis.readInt();
        double wd = dis.readDouble();
        int count = dis.readInt();
        mW = new ArrayList<>(); vW = new ArrayList<>();
        mB = new ArrayList<>(); vB = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            mW.add(ml.network.ModelSerializer.readDoubles(dis));
            vW.add(ml.network.ModelSerializer.readDoubles(dis));
            mB.add(ml.network.ModelSerializer.readDoubles(dis));
            vB.add(ml.network.ModelSerializer.readDoubles(dis));
        }
        initialized = true;
    }
}
