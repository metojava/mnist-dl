package ml.augmentation;

import ml.layer.Layer;
import ml.core.Tensor;

import java.util.Random;

public class RandomRotation implements Layer {
    private final double maxAngle;
    private final double probability;
    private final Random rng;
    private double[] rotationAngles;

    public RandomRotation(double maxAngleDegrees, double probability, long seed) {
        this.maxAngle = Math.toRadians(maxAngleDegrees);
        this.probability = probability;
        this.rng = new Random(seed);
    }

    public RandomRotation(double maxAngleDegrees, double probability) {
        this(maxAngleDegrees, probability, System.nanoTime());
    }

    public RandomRotation(double maxAngleDegrees) {
        this(maxAngleDegrees, 0.5);
    }

    public RandomRotation() {
        this(15.0, 0.5);
    }

    public double getMaxAngleDegrees() { return Math.toDegrees(maxAngle); }
    public double getProbability() { return probability; }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int batchSize = input.rows();
        int features = input.cols();

        if (!training) {
            rotationAngles = new double[batchSize];
            return input;
        }

        int channels = detectChannels(features);
        int spatialSize = features / channels;
        int side = (int) Math.sqrt(spatialSize);

        if (side * side != spatialSize) {
            rotationAngles = new double[batchSize];
            return input;
        }

        rotationAngles = new double[batchSize];
        Tensor output = Tensor.zeros(batchSize, features);
        double center = (side - 1) / 2.0;

        for (int b = 0; b < batchSize; b++) {
            boolean doRotate = rng.nextDouble() < probability;
            double angle = doRotate ? (rng.nextDouble() * 2 - 1) * maxAngle : 0;
            rotationAngles[b] = angle;

            if (!doRotate) {
                System.arraycopy(input.data(), b * features, output.data(), b * features, features);
                continue;
            }

            double cosA = Math.cos(angle);
            double sinA = Math.sin(angle);

            for (int c = 0; c < channels; c++) {
                int channelOffset = c * spatialSize;
                for (int dstY = 0; dstY < side; dstY++) {
                    for (int dstX = 0; dstX < side; dstX++) {
                        double srcXD = (dstX - center) * cosA + (dstY - center) * sinA + center;
                        double srcYD = -(dstX - center) * sinA + (dstY - center) * cosA + center;

                        int srcX0 = (int) Math.floor(srcXD);
                        int srcY0 = (int) Math.floor(srcYD);
                        int srcX1 = srcX0 + 1;
                        int srcY1 = srcY0 + 1;

                        double wx = srcXD - srcX0;
                        double wy = srcYD - srcY0;

                        double val = 0;
                        if (srcX0 >= 0 && srcX0 < side && srcY0 >= 0 && srcY0 < side) {
                            val += (1 - wx) * (1 - wy) * input.get(0, b * features + channelOffset + srcY0 * side + srcX0);
                        }
                        if (srcX1 >= 0 && srcX1 < side && srcY0 >= 0 && srcY0 < side) {
                            val += wx * (1 - wy) * input.get(0, b * features + channelOffset + srcY0 * side + srcX1);
                        }
                        if (srcX0 >= 0 && srcX0 < side && srcY1 >= 0 && srcY1 < side) {
                            val += (1 - wx) * wy * input.get(0, b * features + channelOffset + srcY1 * side + srcX0);
                        }
                        if (srcX1 >= 0 && srcX1 < side && srcY1 >= 0 && srcY1 < side) {
                            val += wx * wy * input.get(0, b * features + channelOffset + srcY1 * side + srcX1);
                        }

                        output.data()[b * features + channelOffset + dstY * side + dstX] = (float) val;
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
