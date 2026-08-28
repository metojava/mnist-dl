package ml.augmentation;

import ml.layer.Layer;
import ml.core.Tensor;

import java.util.Random;

public class RandomFlip implements Layer {
    private final String mode;
    private final double probability;
    private final Random rng;
    private boolean[] flipApplied;

    public RandomFlip(String mode, double probability, long seed) {
        this.mode = mode.toLowerCase();
        this.probability = probability;
        this.rng = new Random(seed);
        if (!mode.equals("horizontal") && !mode.equals("vertical") && !mode.equals("both")) {
            throw new IllegalArgumentException("mode must be 'horizontal', 'vertical', or 'both'");
        }
    }

    public RandomFlip(String mode, double probability) {
        this(mode, probability, System.nanoTime());
    }

    public RandomFlip(String mode) {
        this(mode, 0.5);
    }

    public RandomFlip() {
        this("horizontal", 0.5);
    }

    public String getMode() { return mode; }
    public double getProbability() { return probability; }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int batchSize = input.rows();
        int features = input.cols();

        if (!training) {
            flipApplied = new boolean[batchSize];
            return input;
        }

        int channels = detectChannels(features);
        int spatialSize = features / channels;
        int side = (int) Math.sqrt(spatialSize);

        if (side * side != spatialSize) {
            flipApplied = new boolean[batchSize];
            return input;
        }

        flipApplied = new boolean[batchSize];
        Tensor output = new Tensor(batchSize, features);

        for (int b = 0; b < batchSize; b++) {
            boolean doFlip = rng.nextDouble() < probability;
            flipApplied[b] = doFlip;

            if (!doFlip) {
                System.arraycopy(input.data(), b * features, output.data(), b * features, features);
                continue;
            }

            for (int c = 0; c < channels; c++) {
                int channelOffset = c * spatialSize;
                for (int y = 0; y < side; y++) {
                    for (int x = 0; x < side; x++) {
                        int srcIdx = b * features + channelOffset + y * side + x;
                        int dstX, dstY;

                        if (mode.equals("horizontal") || mode.equals("both")) {
                            dstX = side - 1 - x;
                        } else {
                            dstX = x;
                        }

                        if (mode.equals("vertical") || mode.equals("both")) {
                            dstY = side - 1 - y;
                        } else {
                            dstY = y;
                        }

                        int dstIdx = b * features + channelOffset + dstY * side + dstX;
                        output.data()[dstIdx] = input.data()[srcIdx];
                    }
                }
            }
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradient) {
        if (flipApplied == null) return gradient;

        int batchSize = gradient.rows();
        int features = gradient.cols();
        int channels = detectChannels(features);
        int spatialSize = features / channels;
        int side = (int) Math.sqrt(spatialSize);

        Tensor output = new Tensor(batchSize, features);

        for (int b = 0; b < batchSize; b++) {
            if (!flipApplied[b]) {
                System.arraycopy(gradient.data(), b * features, output.data(), b * features, features);
                continue;
            }

            for (int c = 0; c < channels; c++) {
                int channelOffset = c * spatialSize;
                for (int y = 0; y < side; y++) {
                    for (int x = 0; x < side; x++) {
                        int srcIdx = b * features + channelOffset + y * side + x;
                        int dstX, dstY;

                        if (mode.equals("horizontal") || mode.equals("both")) {
                            dstX = side - 1 - x;
                        } else {
                            dstX = x;
                        }

                        if (mode.equals("vertical") || mode.equals("both")) {
                            dstY = side - 1 - y;
                        } else {
                            dstY = y;
                        }

                        int dstIdx = b * features + channelOffset + dstY * side + dstX;
                        output.data()[dstIdx] = gradient.data()[srcIdx];
                    }
                }
            }
        }
        return output;
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
