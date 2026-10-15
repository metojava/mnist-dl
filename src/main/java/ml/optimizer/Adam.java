package ml.optimizer;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import ml.core.ThreadPoolManager;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Adam implements Optimizer {
    private double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private int timestep;
    private List<double[]> mW, vW;
    private List<double[]> mB, vB;
    private boolean initialized;

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.timestep = 0;
        this.initialized = false;
    }

    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam() {
        this(0.001);
    }

    @Override
    public void step(List<Layer> layers, double lr) {
        double effectiveLr = lr > 0 ? lr : this.learningRate;
        if (!initialized) init(layers);
        timestep++;
        double bc1 = 1.0 - Math.pow(beta1, timestep);
        double bc2 = 1.0 - Math.pow(beta2, timestep);
        int denseIdx = 0;
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer dense) {
                if (dense.weightsGradient() == null) { denseIdx++; continue; }
                final int idxCapture = denseIdx++;
                int wLen = dense.weights().size();
                float[] wData = dense.weights().data();
                float[] wgData = dense.weightsGradient().data();
                double[] mWData = mW.get(idxCapture);
                double[] vWData = vW.get(idxCapture);
                double invBc1 = 1.0 / bc1;
                double invBc2 = 1.0 / bc2;

                ThreadPoolManager.parallelFor(0, wLen, i -> {
                    double g = wgData[i];
                    mWData[i] = beta1 * mWData[i] + (1 - beta1) * g;
                    vWData[i] = beta2 * vWData[i] + (1 - beta2) * g * g;
                    double mHat = mWData[i] * invBc1;
                    double vHat = vWData[i] * invBc2;
                    wData[i] -= effectiveLr * mHat / (Math.sqrt(vHat) + epsilon);
                });

                int bLen = dense.bias().size();
                float[] bData = dense.bias().data();
                float[] bgData = dense.biasGradient().data();
                double[] mBData = mB.get(idxCapture);
                double[] vBData = vB.get(idxCapture);

                for (int i = 0; i < bLen; i++) {
                    double g = bgData[i];
                    mBData[i] = beta1 * mBData[i] + (1 - beta1) * g;
                    vBData[i] = beta2 * vBData[i] + (1 - beta2) * g * g;
                    double mHat = mBData[i] * invBc1;
                    double vHat = vBData[i] * invBc2;
                    bData[i] -= effectiveLr * mHat / (Math.sqrt(vHat) + epsilon);
                }
                dense.invalidateTranspose();
            } else {
                // Self-updating layers (Conv2D, LSTM, Embedding, LayerNorm, ...)
                layer.update(effectiveLr);
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
