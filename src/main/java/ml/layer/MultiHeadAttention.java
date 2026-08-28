package ml.layer;

import ml.core.Tensor;

public class MultiHeadAttention implements Layer {
    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final boolean isCrossAttention;

    private final Tensor[] wq;
    private final Tensor[] wk;
    private final Tensor[] wv;
    private final Tensor wo;

    private Tensor lastInput;
    private Tensor lastKey;
    private Tensor lastValue;
    private Tensor lastAttnWeights;

    public MultiHeadAttention(int embedDim, int numHeads, boolean isCrossAttention) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.isCrossAttention = isCrossAttention;

        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }

        this.wq = new Tensor[numHeads];
        this.wk = new Tensor[numHeads];
        this.wv = new Tensor[numHeads];
        for (int i = 0; i < numHeads; i++) {
            this.wq[i] = Tensor.he(headDim, embedDim);
            this.wk[i] = Tensor.he(headDim, embedDim);
            this.wv[i] = Tensor.he(headDim, embedDim);
        }
        this.wo = Tensor.he(embedDim, embedDim);
    }

    public MultiHeadAttention(int embedDim, int numHeads) {
        this(embedDim, numHeads, false);
    }

    public int getEmbedDim() { return embedDim; }
    public int getNumHeads() { return numHeads; }
    public boolean isCrossAttention() { return isCrossAttention; }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        return forward(input, input, training);
    }

    public Tensor forward(Tensor query, Tensor keyValue, boolean training) {
        int seqLenQ = query.cols() / embedDim;
        int seqLenKV = keyValue.cols() / embedDim;
        int batchSize = query.rows();
        this.lastInput = query;
        this.lastKey = keyValue;
        this.lastValue = keyValue;

        Tensor[] headOutputs = new Tensor[numHeads];
        this.lastAttnWeights = null;

        for (int h = 0; h < numHeads; h++) {
            Tensor q = Tensor.matmul(query, wq[h].transpose());
            Tensor k = Tensor.matmul(keyValue, wk[h].transpose());
            Tensor v = Tensor.matmul(keyValue, wv[h].transpose());

            double scale = Math.sqrt(headDim);
            Tensor scores = Tensor.matmul(q, k.transpose());
            scores = scores.multiply(1.0 / scale);

            Tensor attnWeights = softmaxSeq(scores, seqLenQ, seqLenKV);
            if (h == 0) {
                this.lastAttnWeights = attnWeights;
            }

            headOutputs[h] = Tensor.matmul(attnWeights, v);
        }

        Tensor concatenated = concatHeads(headOutputs, batchSize);

        Tensor output = Tensor.matmul(concatenated, wo.transpose());
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
        int params = 0;
        for (int h = 0; h < numHeads; h++) {
            params += wq[h].size() + wk[h].size() + wv[h].size();
        }
        params += wo.size();
        return params;
    }

    private Tensor softmaxSeq(Tensor input, int seqLenQ, int seqLenKV) {
        Tensor result = new Tensor(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < seqLenKV; j++) {
                double val = input.get(i, j);
                if (val > max) max = val;
            }
            double sum = 0;
            for (int j = 0; j < seqLenKV; j++) {
                double val = Math.exp(input.get(i, j) - max);
                result.data()[i * input.cols() + j] = val;
                sum += val;
            }
            for (int j = 0; j < seqLenKV; j++) {
                result.data()[i * input.cols() + j] /= sum;
            }
        }
        return result;
    }

    private Tensor concatHeads(Tensor[] heads, int batchSize) {
        int seqLen = heads[0].cols() / headDim;
        Tensor result = Tensor.zeros(batchSize, seqLen * embedDim);
        for (int h = 0; h < numHeads; h++) {
            for (int t = 0; t < seqLen; t++) {
                for (int d = 0; d < headDim; d++) {
                    result.data()[0 * result.cols() + t * embedDim + h * headDim + d] =
                        heads[h].get(0, t * headDim + d);
                }
            }
        }
        return result;
    }

    public Tensor getAttentionWeights() { return lastAttnWeights; }
    public Tensor[] weightsWq() { return wq; }
    public Tensor[] weightsWk() { return wk; }
    public Tensor[] weightsWv() { return wv; }
    public Tensor weightsWo() { return wo; }
}
