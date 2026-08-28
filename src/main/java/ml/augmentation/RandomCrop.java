package ml.augmentation;

import ml.layer.Layer;
import ml.core.Tensor;

import java.util.Random;

public class RandomCrop implements Layer {
    private final int cropSize;
    private final double probability;
    private final Random rng;
    private int[][] cropOffsets;

    public RandomCrop(int cropSize, double probability, long seed) {
        this.cropSize = cropSize;
        this.probability = probability;
        this.rng = new Random(seed);
    }

    public RandomCrop(int cropSize, double probability) {
        this(cropSize, probability, System.nanoTime());
    }

    public RandomCrop(int cropSize) {
        this(cropSize, 1.0);
    }

    public int getCropSize() { return cropSize; }
    public double getProbability() { return probability; }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int batchSize = input.rows();
        int features = input.cols();

        if (!training) {
            cropOffsets = new int[batchSize][2];
            return input;
        }

        int channels = detectChannels(features);
        int spatialSize = features / channels;
        int origSide = (int) Math.sqrt(spatialSize);

        if (origSide * origSide != spatialSize || cropSize > origSide) {
            cropOffsets = new int[batchSize][2];
            return input;
        }

        cropOffsets = new int[batchSize][2];
        int newFeatures = channels * cropSize * cropSize;
        Tensor output = Tensor.zeros(batchSize, newFeatures);

        for (int b = 0; b < batchSize; b++) {
            boolean doCrop = rng.nextDouble() < probability;

            if (!doCrop) {
                int offsetX = (origSide - cropSize) / 2;
                int offsetY = (origSide - cropSize) / 2;
                cropOffsets[b] = new int[]{offsetX, offsetY};

                for (int c = 0; c < channels; c++) {
                    for (int y = 0; y < cropSize; y++) {
                        for (int x = 0; x < cropSize; x++) {
                            int srcIdx = b * features + c * spatialSize + (y + offsetY) * origSide + (x + offsetX);
                            int dstIdx = b * newFeatures + c * cropSize * cropSize + y * cropSize + x;
                            output.data()[dstIdx] = input.data()[srcIdx];
                        }
                    }
                }
            } else {
                int offsetX = rng.nextInt(origSide - cropSize + 1);
                int offsetY = rng.nextInt(origSide - cropSize + 1);
                cropOffsets[b] = new int[]{offsetX, offsetY};

                for (int c = 0; c < channels; c++) {
                    for (int y = 0; y < cropSize; y++) {
                        for (int x = 0; x < cropSize; x++) {
                            int srcIdx = b * features + c * spatialSize + (y + offsetY) * origSide + (x + offsetX);
                            int dstIdx = b * newFeatures + c * cropSize * cropSize + y * cropSize + x;
                            output.data()[dstIdx] = input.data()[srcIdx];
                        }
                    }
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

    private int detectChannels(int features) {
        int side = (int) Math.sqrt(features);
        if (side * side == features) return 1;
        for (int c = 3; c <= 4; c++) {
            int spatial = features / c;
            int s = (int) Math.sqrt(spatial);
            if (s * s == spatial) return c;
        }
        return 1;
    }
}
