package ml.layer;

import ml.core.Tensor;

public class SelfAttention implements Layer {
    private final int embedDim;
    private final boolean scaled;
    private final double scale;

    private final Tensor wq;
    private final Tensor wk;
    private final Tensor wv;
    private final Tensor wo;

    private Tensor wqGrad, wkGrad, wvGrad, woGrad;

    private Tensor lastInput;
    private Tensor lastQ, lastK, lastV;
    private Tensor lastAttnWeights;

    public SelfAttention(int embedDim, boolean scaled) {
        this.embedDim = embedDim;
        this.scaled = scaled;
        this.scale = scaled ? Math.sqrt(embedDim) : 1.0;

        this.wq = Tensor.he(embedDim, embedDim);
        this.wk = Tensor.he(embedDim, embedDim);
        this.wv = Tensor.he(embedDim, embedDim);
        this.wo = Tensor.he(embedDim, embedDim);
    }

    public SelfAttention(int embedDim) {
        this(embedDim, true);
    }

    public int getEmbedDim() { return embedDim; }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int seqLen = input.cols() / embedDim;
        int batchSize = input.rows();
        this.lastInput = input;

        this.lastQ = Tensor.matmul(input, wq.transpose());
        this.lastK = Tensor.matmul(input, wk.transpose());
        this.lastV = Tensor.matmul(input, wv.transpose());

        Tensor scores = Tensor.matmul(lastQ, lastK.transpose());
        if (scaled) {
            scores = scores.multiply(1.0 / scale);
        }

        this.lastAttnWeights = softmaxLastDim(scores, seqLen);

        Tensor context = Tensor.matmul(lastAttnWeights, lastV);

        Tensor output = Tensor.matmul(context, wo.transpose());
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        return gradient;
    }

    @Override
    public void update(double learningRate) {}

    @Override
    public int inputSize() { return embedDim; }

    @Override
    public int outputSize() { return embedDim; }

    @Override
    public int parameterCount() {
        return wq.size() + wk.size() + wv.size() + wo.size();
    }

    private Tensor softmaxLastDim(Tensor input, int seqLen) {
        Tensor result = new Tensor(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < seqLen; j++) {
                double val = input.get(i, j);
                if (val > max) max = val;
            }
            double sum = 0;
            for (int j = 0; j < seqLen; j++) {
                double val = Math.exp(input.get(i, j) - max);
                result.data()[i * input.cols() + j] = val;
                sum += val;
            }
            for (int j = 0; j < seqLen; j++) {
                result.data()[i * input.cols() + j] /= sum;
            }
        }
        return result;
    }

    public Tensor getAttentionWeights() { return lastAttnWeights; }
    public Tensor weightsWq() { return wq; }
    public Tensor weightsWk() { return wk; }
    public Tensor weightsWv() { return wv; }
    public Tensor weightsWo() { return wo; }
}
