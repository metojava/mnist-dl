package ml.layer;

import ml.core.Tensor;

public class AvgPool implements Layer {
    private final int poolSize;
    private final int stride;
    private int inputH;
    private int inputW;
    private int outH;
    private int outW;

    public AvgPool(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    public AvgPool(int poolSize) {
        this(poolSize, poolSize);
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int batch = input.rows();
        inputH = (int) Math.sqrt(input.cols());
        inputW = inputH;
        outH = (inputH - poolSize) / stride + 1;
        outW = (inputW - poolSize) / stride + 1;
        int outSize = outH * outW;
        Tensor output = Tensor.zeros(batch, outSize);
        double invArea = 1.0 / (poolSize * poolSize);
        for (int b = 0; b < batch; b++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    double sum = 0;
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            if (ih < inputH && iw < inputW) {
                                sum += input.data()[b * input.cols() + ih * inputW + iw];
                            }
                        }
                    }
                    output.data()[b * outSize + oh * outW + ow] = sum * invArea;
                }
            }
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        return gradient;
    }

    @Override
    public void update(double learningRate) {}

    @Override
    public int inputSize() { return -1; }
    @Override
    public int outputSize() { return -1; }
    @Override
    public int parameterCount() { return 0; }
}
