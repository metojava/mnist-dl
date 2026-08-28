package ml.layer;

import ml.core.Tensor;

public class MaxPool implements Layer {
    private final int poolSize;
    private final int stride;
    private final int channels;
    private Tensor lastInput;
    private int inputH;
    private int inputW;
    private int outH;
    private int outW;
    private int[] argmax;

    public MaxPool(int poolSize, int stride, int channels) {
        this.poolSize = poolSize;
        this.stride = stride;
        this.channels = channels;
    }

    public MaxPool(int poolSize, int stride) {
        this(poolSize, stride, 1);
    }

    public MaxPool(int poolSize) {
        this(poolSize, poolSize, 1);
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        lastInput = input;
        int batch = input.rows();
        inputH = (int) Math.sqrt(input.cols() / channels);
        inputW = inputH;
        outH = (inputH - poolSize) / stride + 1;
        outW = (inputW - poolSize) / stride + 1;
        int outSize = channels * outH * outW;
        Tensor output = Tensor.zeros(batch, outSize);
        argmax = new int[batch * outSize];

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                int inChannelOffset = b * input.cols() + c * inputH * inputW;
                int outChannelOffset = b * outSize + c * outH * outW;
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double maxVal = Double.NEGATIVE_INFINITY;
                        int maxIdx = inChannelOffset;
                        for (int ph = 0; ph < poolSize; ph++) {
                            for (int pw = 0; pw < poolSize; pw++) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                if (ih < inputH && iw < inputW) {
                                    int idx = inChannelOffset + ih * inputW + iw;
                                    double v = input.data()[idx];
                                    if (v > maxVal) { maxVal = v; maxIdx = idx; }
                                }
                            }
                        }
                        output.data()[outChannelOffset + oh * outW + ow] = maxVal;
                        argmax[outChannelOffset + oh * outW + ow] = maxIdx;
                    }
                }
            }
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        if (argmax == null) return gradient;
        int batch = lastInput.rows();
        int outSize = channels * outH * outW;
        Tensor dInput = Tensor.zeros(batch, lastInput.cols());
        for (int i = 0; i < batch * outSize; i++) {
            dInput.data()[argmax[i]] += gradient.data()[i];
        }
        return dInput;
    }

    @Override
    public void update(double learningRate) {}

    @Override
    public int inputSize() { return -1; }
    @Override
    public int outputSize() { return -1; }
    @Override
    public int parameterCount() { return 0; }

    public int pooledOutputSize() { return channels * outH * outW; }
}
