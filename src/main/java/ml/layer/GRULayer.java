package ml.layer;

import ml.core.Tensor;

import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;

public class GRULayer implements Layer {
    private final int inputSize;
    private final int hiddenSize;
    private final boolean bidirectional;
    private final boolean returnSequences;

    private final Tensor wxz;
    private final Tensor whz;
    private final Tensor bhz;
    private final Tensor wxr;
    private final Tensor whr;
    private final Tensor bhr;
    private final Tensor wxh;
    private final Tensor whh;
    private final Tensor bhh;

    private Tensor wxzGrad, whzGrad, bhzGrad;
    private Tensor wxrGrad, whrGrad, bhrGrad;
    private Tensor wxhGrad, whhGrad, bhhGrad;
    private final ReentrantLock gradLock = new ReentrantLock();

    private Tensor lastInput;

    public GRULayer(int inputSize, int hiddenSize, boolean bidirectional, boolean returnSequences) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.bidirectional = bidirectional;
        this.returnSequences = returnSequences;

        this.wxz = Tensor.he(hiddenSize, inputSize);
        this.whz = Tensor.he(hiddenSize, hiddenSize);
        this.bhz = Tensor.zeros(1, hiddenSize);

        this.wxr = Tensor.he(hiddenSize, inputSize);
        this.whr = Tensor.he(hiddenSize, hiddenSize);
        this.bhr = Tensor.zeros(1, hiddenSize);

        this.wxh = Tensor.he(hiddenSize, inputSize);
        this.whh = Tensor.he(hiddenSize, hiddenSize);
        this.bhh = Tensor.zeros(1, hiddenSize);
    }

    public GRULayer(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, false, false);
    }

    public GRULayer(int inputSize, int hiddenSize, boolean bidirectional) {
        this(inputSize, hiddenSize, bidirectional, false);
    }

    public int getInputSize() { return inputSize; }
    public int getHiddenSize() { return hiddenSize; }
    public boolean isBidirectional() { return bidirectional; }
    public boolean isReturnSequences() { return returnSequences; }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, x))));
    }

    private double[] forwardStep(double[] x_t, double[] hPrev) {
        double[] zGate = new double[hiddenSize];
        double[] rGate = new double[hiddenSize];
        double[] hHat = new double[hiddenSize];

        for (int j = 0; j < hiddenSize; j++) {
            double sumZ = bhz.get(0, j);
            double sumR = bhr.get(0, j);
            double sumH = bhh.get(0, j);

            for (int i = 0; i < inputSize; i++) {
                sumZ += wxz.get(j, i) * x_t[i];
                sumR += wxr.get(j, i) * x_t[i];
                sumH += wxh.get(j, i) * x_t[i];
            }
            for (int k = 0; k < hiddenSize; k++) {
                sumZ += whz.get(j, k) * hPrev[k];
                sumR += whr.get(j, k) * hPrev[k];
            }

            zGate[j] = sigmoid(sumZ);
            rGate[j] = sigmoid(sumR);

            for (int k = 0; k < hiddenSize; k++) {
                sumH += whh.get(j, k) * rGate[j] * hPrev[k];
            }
            hHat[j] = Math.tanh(sumH);
        }

        double[] hNew = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            hNew[j] = (1 - zGate[j]) * hHat[j] + zGate[j] * hPrev[j];
        }

        return hNew;
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int seqLen = input.cols() / inputSize;
        int batchSize = input.rows();
        this.lastInput = input;

        double[][] hForward = new double[seqLen + 1][];
        hForward[0] = new double[hiddenSize];

        for (int t = 0; t < seqLen; t++) {
            double[] x_t = new double[inputSize];
            for (int i = 0; i < inputSize; i++) {
                x_t[i] = input.get(0, t * inputSize + i);
            }
            hForward[t + 1] = forwardStep(x_t, hForward[t]);
        }

        if (!bidirectional) {
            if (returnSequences) {
                Tensor output = Tensor.zeros(batchSize, seqLen * hiddenSize);
                for (int t = 0; t < seqLen; t++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        output.set(0, t * hiddenSize + j, hForward[t + 1][j]);
                    }
                }
                return output;
            } else {
                Tensor output = Tensor.zeros(batchSize, hiddenSize);
                for (int j = 0; j < hiddenSize; j++) {
                    output.set(0, j, hForward[seqLen][j]);
                }
                return output;
            }
        } else {
            double[][] hBackward = new double[seqLen + 1][];
            hBackward[seqLen] = new double[hiddenSize];

            for (int t = seqLen - 1; t >= 0; t--) {
                double[] x_t = new double[inputSize];
                for (int i = 0; i < inputSize; i++) {
                    x_t[i] = input.get(0, t * inputSize + i);
                }
                hBackward[t] = forwardStep(x_t, hBackward[t + 1]);
            }

            int effectiveOut = hiddenSize * 2;
            if (returnSequences) {
                Tensor output = Tensor.zeros(batchSize, seqLen * effectiveOut);
                for (int t = 0; t < seqLen; t++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        output.set(0, t * effectiveOut + j, hForward[t + 1][j]);
                        output.set(0, t * effectiveOut + hiddenSize + j, hBackward[t][j]);
                    }
                }
                return output;
            } else {
                Tensor output = Tensor.zeros(batchSize, effectiveOut);
                for (int j = 0; j < hiddenSize; j++) {
                    output.set(0, j, hForward[seqLen][j]);
                    output.set(0, hiddenSize + j, hBackward[0][j]);
                }
                return output;
            }
        }
    }

    @Override
    public Tensor backward(Tensor gradient) {
        return gradient;
    }

    @Override
    public void update(double learningRate) {}

    @Override
    public int inputSize() { return inputSize; }

    @Override
    public int outputSize() {
        int effectiveOut = bidirectional ? hiddenSize * 2 : hiddenSize;
        return returnSequences ? -1 : effectiveOut;
    }

    @Override
    public int parameterCount() {
        int params = wxz.size() + whz.size() + bhz.size();
        params += wxr.size() + whr.size() + bhr.size();
        params += wxh.size() + whh.size() + bhh.size();
        if (bidirectional) params *= 2;
        return params;
    }

    public Tensor weightsWxz() { return wxz; }
    public Tensor weightsWhz() { return whz; }
    public Tensor biasHz() { return bhz; }
    public Tensor weightsWxr() { return wxr; }
    public Tensor weightsWhr() { return whr; }
    public Tensor biasHr() { return bhr; }
    public Tensor weightsWxh() { return wxh; }
    public Tensor weightsWhh() { return whh; }
    public Tensor biasHh() { return bhh; }
}
