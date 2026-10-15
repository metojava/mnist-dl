package ml.layer;

import ml.core.Tensor;
import ml.exception.ShapeException;
import java.util.Random;

public class Conv2D implements Layer {
    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final int outH;
    private final int outW;
    private final Tensor[] kernels;
    private final double[] biases;
    private final Random rng;
    private Tensor lastInput;
    private int inputH;
    private int inputW;
    private final double[] kernelGrads;
    private final double[] biasGrads;
    private int lastBatch;
    private int lastH;
    private int lastW;

    public Conv2D(int inChannels, int outChannels, int kernelSize, int stride, int padding, Random rng) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.rng = rng;
        this.kernels = new Tensor[outChannels];
        this.biases = new double[outChannels];
        this.kernelGrads = new double[outChannels * inChannels * kernelSize * kernelSize];
        this.biasGrads = new double[outChannels];
        this.outH = -1;
        this.outW = -1;
        double std = Math.sqrt(2.0 / (inChannels * kernelSize * kernelSize));
        for (int i = 0; i < outChannels; i++) {
            kernels[i] = Tensor.randomNormal(inChannels * kernelSize * kernelSize, 1, rng).apply(x -> x * std);
        }
    }

    public Conv2D(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0, new Random());
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        lastInput = input;
        int batch = input.rows();
        int channels = inChannels;
        inputH = (int) Math.sqrt(input.cols() / channels);
        inputW = inputH;
        int h = (inputH + 2 * padding - kernelSize) / stride + 1;
        int w = (inputW + 2 * padding - kernelSize) / stride + 1;
        lastBatch = batch;
        lastH = h;
        lastW = w;
        int outSize = outChannels * h * w;
        Tensor output = Tensor.zeros(batch, outSize);
        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outChannels; oc++) {
                for (int oh = 0; oh < h; oh++) {
                    for (int ow = 0; ow < w; ow++) {
                        double sum = biases[oc];
                        for (int ic = 0; ic < inChannels; ic++) {
                            for (int kh = 0; kh < kernelSize; kh++) {
                                for (int kw = 0; kw < kernelSize; kw++) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW) {
                                        int inIdx = b * input.cols() + ic * inputH * inputW + ih * inputW + iw;
                                        int kIdx = ic * kernelSize * kernelSize + kh * kernelSize + kw;
                                        sum += input.data()[inIdx] * kernels[oc].data()[kIdx];
                                    }
                                }
                            }
                        }
                        int outIdx = b * outSize + oc * h * w + oh * w + ow;
                        output.data()[outIdx] = (float) sum;
                    }
                }
            }
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        int batch = lastBatch;
        int h = lastH;
        int w = lastW;
        int outSize = outChannels * h * w;

        Tensor dInput = Tensor.zeros(batch, lastInput.cols());
        java.util.Arrays.fill(kernelGrads, 0.0);
        java.util.Arrays.fill(biasGrads, 0.0);

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outChannels; oc++) {
                for (int oh = 0; oh < h; oh++) {
                    for (int ow = 0; ow < w; ow++) {
                        double g = gradient.data()[b * outSize + oc * h * w + oh * w + ow];
                        biasGrads[oc] += g;
                        for (int ic = 0; ic < inChannels; ic++) {
                            for (int kh = 0; kh < kernelSize; kh++) {
                                for (int kw = 0; kw < kernelSize; kw++) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW) {
                                        int inIdx = b * lastInput.cols() + ic * inputH * inputW + ih * inputW + iw;
                                        int kLocal = ic * kernelSize * kernelSize + kh * kernelSize + kw;
                                        int kGrad = oc * inChannels * kernelSize * kernelSize + kLocal;
                                        dInput.data()[inIdx] += g * kernels[oc].data()[kLocal];
                                        kernelGrads[kGrad] += g * lastInput.data()[inIdx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return dInput;
    }

    @Override
    public void update(double learningRate) {
        double scale = learningRate / Math.max(1, lastBatch);
        for (int oc = 0; oc < outChannels; oc++) {
            float[] kd = kernels[oc].data();
            int gradBase = oc * kd.length;
            for (int i = 0; i < kd.length; i++) {
                kd[i] -= (float) (scale * kernelGrads[gradBase + i]);
                kernelGrads[gradBase + i] = 0.0;
            }
            biases[oc] -= scale * biasGrads[oc];
            biasGrads[oc] = 0.0;
        }
    }

    @Override
    public int inputSize() { return inChannels; }
    @Override
    public int outputSize() { return outChannels; }
    @Override
    public int parameterCount() {
        int total = 0;
        for (Tensor k : kernels) total += k.size();
        total += outChannels;
        return total;
    }

    public Tensor[] kernels() { return kernels; }
    public double[] biases() { return biases; }
    public int kernelCount() { return outChannels; }
    public Tensor kernel(int i) { return kernels[i]; }
    public int kernelSize() { return kernelSize; }
    public int stride() { return stride; }
    public int padding() { return padding; }
    public Tensor bias() {
        Tensor b = Tensor.zeros(1, outChannels);
        for (int i = 0; i < outChannels; i++) b.data()[i] = (float) biases[i];
        return b;
    }
}
