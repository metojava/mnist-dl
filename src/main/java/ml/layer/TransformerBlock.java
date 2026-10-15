package ml.layer;

import ml.core.Tensor;
import ml.regularization.Dropout;

public class TransformerBlock implements Layer {
    private final int embedDim;
    private final int numHeads;
    private final int feedForwardDim;
    private final boolean isCrossAttention;
    private final boolean preNorm;

    private final MultiHeadAttention attention;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final LayerNorm norm3;
    private final DenseLayer ff1;
    private final DenseLayer ff2;
    private final Dropout dropout1;
    private final Dropout dropout2;
    private final Dropout dropout3;

    private Tensor lastInput;

    public TransformerBlock(int embedDim, int numHeads, int feedForwardDim,
                            boolean isCrossAttention, boolean preNorm, double dropoutRate) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.feedForwardDim = feedForwardDim;
        this.isCrossAttention = isCrossAttention;
        this.preNorm = preNorm;

        this.attention = new MultiHeadAttention(embedDim, numHeads, isCrossAttention);
        this.norm1 = new LayerNorm(embedDim);
        this.norm2 = new LayerNorm(embedDim);
        this.norm3 = isCrossAttention ? new LayerNorm(embedDim) : null;
        this.ff1 = new DenseLayer(embedDim, feedForwardDim);
        this.ff2 = new DenseLayer(feedForwardDim, embedDim);
        this.dropout1 = new Dropout(dropoutRate);
        this.dropout2 = new Dropout(dropoutRate);
        this.dropout3 = isCrossAttention ? new Dropout(dropoutRate) : null;
    }

    public TransformerBlock(int embedDim, int numHeads, int feedForwardDim) {
        this(embedDim, numHeads, feedForwardDim, false, true, 0.1);
    }

    public TransformerBlock(int embedDim, int numHeads, int feedForwardDim, boolean isCrossAttention) {
        this(embedDim, numHeads, feedForwardDim, isCrossAttention, true, 0.1);
    }

    public TransformerBlock(int embedDim, int numHeads, int feedForwardDim, double dropoutRate) {
        this(embedDim, numHeads, feedForwardDim, false, true, dropoutRate);
    }

    public int getEmbedDim() { return embedDim; }
    public int getNumHeads() { return numHeads; }
    public int getFeedForwardDim() { return feedForwardDim; }
    public boolean isCrossAttention() { return isCrossAttention; }
    public boolean isPreNorm() { return preNorm; }

    public Tensor forward(Tensor input, boolean training) {
        return forward(input, null, training);
    }

    public Tensor forward(Tensor input, Tensor encoderOutput, boolean training) {
        this.lastInput = input;
        Tensor x = input;

        if (preNorm) {
            Tensor normed = norm1.forward(x, training);
            Tensor attnOut;
            if (isCrossAttention && encoderOutput != null) {
                Tensor normedKV = norm3.forward(encoderOutput, training);
                attnOut = attention.forward(normed, normedKV, training);
            } else {
                attnOut = attention.forward(normed, training);
            }
            attnOut = dropout1.forward(attnOut, training);
            x = residualAdd(x, attnOut);

            Tensor normed2 = norm2.forward(x, training);
            Tensor ffOut = ff1.forward(normed2, training);
            ffOut = new ml.activation.ReLU().forward(ffOut);
            ffOut = ff2.forward(ffOut, training);
            ffOut = dropout2.forward(ffOut, training);
            x = residualAdd(x, ffOut);
        } else {
            Tensor attnOut;
            if (isCrossAttention && encoderOutput != null) {
                attnOut = attention.forward(x, encoderOutput, training);
            } else {
                attnOut = attention.forward(x, training);
            }
            attnOut = dropout1.forward(attnOut, training);
            x = residualAdd(x, attnOut);
            x = norm1.forward(x, training);

            Tensor ffOut = ff1.forward(x, training);
            ffOut = new ml.activation.ReLU().forward(ffOut);
            ffOut = ff2.forward(ffOut, training);
            ffOut = dropout2.forward(ffOut, training);
            x = residualAdd(x, ffOut);
            x = norm2.forward(x, training);
        }

        return x;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        return gradient;
    }

    @Override
    public void update(double learningRate) {
        norm1.update(learningRate);
        norm2.update(learningRate);
        if (norm3 != null) norm3.update(learningRate);
        ff1.update(learningRate);
        ff2.update(learningRate);
        dropout1.update(learningRate);
        dropout2.update(learningRate);
        if (dropout3 != null) dropout3.update(learningRate);
    }

    @Override
    public int inputSize() { return embedDim; }

    @Override
    public int outputSize() { return embedDim; }

    @Override
    public int parameterCount() {
        int params = attention.parameterCount();
        params += norm1.parameterCount() + norm2.parameterCount();
        if (norm3 != null) params += norm3.parameterCount();
        params += ff1.parameterCount() + ff2.parameterCount();
        return params;
    }

    private Tensor residualAdd(Tensor a, Tensor b) {
        float[] ad = a.data();
        float[] bd = b.data();
        double[] result = new double[ad.length];
        for (int i = 0; i < ad.length; i++) {
            result[i] = ad[i] + bd[i];
        }
        return Tensor.fromFlat(result, a.rows(), a.cols());
    }

    public MultiHeadAttention getAttention() { return attention; }
    public LayerNorm getNorm1() { return norm1; }
    public LayerNorm getNorm2() { return norm2; }
    public DenseLayer getFf1() { return ff1; }
    public DenseLayer getFf2() { return ff2; }
}
