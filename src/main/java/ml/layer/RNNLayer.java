package ml.layer;

import ml.core.Tensor;
import ml.core.ThreadPoolManager;

import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;

public class RNNLayer implements Layer {
    private final int inputSize;
    private final int outputSize;
    private final boolean bidirectional;
    private final boolean returnSequences;
    private final String activationType;

    private final Tensor wx;
    private final Tensor wh;
    private final Tensor bh;
    private Tensor wxGrad;
    private Tensor whGrad;
    private Tensor bhGrad;
    private final ReentrantLock gradLock = new ReentrantLock();

    private Tensor lastInput;
    private Tensor lastHidden;
    private Tensor[] hiddenStates;

    public RNNLayer(int inputSize, int outputSize, boolean bidirectional, boolean returnSequences, String activationType) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.bidirectional = bidirectional;
        this.returnSequences = returnSequences;
        this.activationType = activationType;

        this.wx = Tensor.he(outputSize, inputSize);
        this.wh = Tensor.he(outputSize, outputSize);
        this.bh = Tensor.zeros(1, outputSize);
    }

    public RNNLayer(int inputSize, int outputSize) {
        this(inputSize, outputSize, false, false, "tanh");
    }

    public RNNLayer(int inputSize, int outputSize, boolean bidirectional) {
        this(inputSize, outputSize, bidirectional, false, "tanh");
    }

    public int getInputSize() { return inputSize; }
    public int getOutputSize() { return outputSize; }
    public boolean isBidirectional() { return bidirectional; }
    public boolean isReturnSequences() { return returnSequences; }

    private double activate(double x) {
        return switch (activationType.toLowerCase()) {
            case "relu" -> Math.max(0, x);
            case "sigmoid" -> 1.0 / (1.0 + Math.exp(-x));
            case "elu" -> x > 0 ? x : Math.exp(x) - 1;
            default -> Math.tanh(x);
        };
    }

    private double activateDerivative(double x, double activated) {
        return switch (activationType.toLowerCase()) {
            case "relu" -> x > 0 ? 1.0 : 0.0;
            case "sigmoid" -> activated * (1 - activated);
            case "elu" -> x > 0 ? 1.0 : activated + 1.0;
            default -> 1 - activated * activated;
        };
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int seqLen = input.cols() / inputSize;
        int batchSize = input.rows();
        this.lastInput = input;

        double[][] allHiddenForward = new double[seqLen + 1][];
        allHiddenForward[0] = new double[outputSize];

        for (int t = 0; t < seqLen; t++) {
            double[] hPrev = allHiddenForward[t];
            double[] x_t = new double[inputSize];
            for (int i = 0; i < inputSize; i++) {
                x_t[i] = input.get(0, t * inputSize + i);
            }

            double[] hNew = new double[outputSize];
            for (int j = 0; j < outputSize; j++) {
                double sum = bh.get(0, j);
                for (int i = 0; i < inputSize; i++) sum += wx.get(j, i) * x_t[i];
                for (int k = 0; k < outputSize; k++) sum += wh.get(j, k) * hPrev[k];
                hNew[j] = activate(sum);
            }
            allHiddenForward[t + 1] = hNew;
        }

        if (!bidirectional) {
            hiddenStates = new Tensor[seqLen + 1];
            for (int t = 0; t <= seqLen; t++) {
                hiddenStates[t] = new Tensor(1, outputSize);
                for (int j = 0; j < outputSize; j++) {
                    hiddenStates[t].set(0, j, allHiddenForward[t][j]);
                }
            }

            if (returnSequences) {
                Tensor output = Tensor.zeros(batchSize, seqLen * outputSize);
                for (int t = 0; t < seqLen; t++) {
                    for (int j = 0; j < outputSize; j++) {
                        output.set(0, t * outputSize + j, allHiddenForward[t + 1][j]);
                    }
                }
                return output;
            } else {
                Tensor output = Tensor.zeros(batchSize, outputSize);
                for (int j = 0; j < outputSize; j++) {
                    output.set(0, j, allHiddenForward[seqLen][j]);
                }
                return output;
            }
        } else {
            double[][] allHiddenBackward = new double[seqLen + 1][];
            allHiddenBackward[seqLen] = new double[outputSize];

            for (int t = seqLen - 1; t >= 0; t--) {
                double[] hNext = allHiddenBackward[t + 1];
                double[] x_t = new double[inputSize];
                for (int i = 0; i < inputSize; i++) {
                    x_t[i] = input.get(0, t * inputSize + i);
                }

                double[] hNew = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    double sum = bh.get(0, j);
                    for (int i = 0; i < inputSize; i++) sum += wx.get(j, i) * x_t[i];
                    for (int k = 0; k < outputSize; k++) sum += wh.get(j, k) * hNext[k];
                    hNew[j] = activate(sum);
                }
                allHiddenBackward[t] = hNew;
            }

            int effectiveOut = outputSize * 2;
            if (returnSequences) {
                Tensor output = Tensor.zeros(batchSize, seqLen * effectiveOut);
                for (int t = 0; t < seqLen; t++) {
                    for (int j = 0; j < outputSize; j++) {
                        output.set(0, t * effectiveOut + j, allHiddenForward[t + 1][j]);
                        output.set(0, t * effectiveOut + outputSize + j, allHiddenBackward[t][j]);
                    }
                }
                return output;
            } else {
                Tensor output = Tensor.zeros(batchSize, effectiveOut);
                for (int j = 0; j < outputSize; j++) {
                    output.set(0, j, allHiddenForward[seqLen][j]);
                    output.set(0, outputSize + j, allHiddenBackward[0][j]);
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
        int effectiveOut = bidirectional ? outputSize * 2 : outputSize;
        return returnSequences ? -1 : effectiveOut;
    }

    @Override
    public int parameterCount() {
        int params = wx.size() + wh.size() + bh.size();
        if (bidirectional) params *= 2;
        return params;
    }

    public Tensor weightsWx() { return wx; }
    public Tensor weightsWh() { return wh; }
    public Tensor biasH() { return bh; }
}
