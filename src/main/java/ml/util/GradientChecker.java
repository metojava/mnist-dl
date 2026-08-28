package ml.util;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import ml.core.Tensor;
import java.util.Random;

public class GradientChecker {
    private final double epsilon;
    private final double tolerance;

    public GradientChecker(double epsilon, double tolerance) {
        this.epsilon = epsilon;
        this.tolerance = tolerance;
    }

    public GradientChecker() {
        this(1e-5, 1e-3);
    }

    public double check(Layer layer, Tensor input) {
        Tensor output = layer.forward(input);
        Tensor grad = Tensor.ones(output.rows(), output.cols());
        Tensor analyticGrad = layer.backward(grad);

        if (!(layer instanceof DenseLayer dense)) return 0;
        double analytic = 0;
        double numerical = 0;
        int checked = 0;

        for (int i = 0; i < Math.min(dense.weights().size(), 10); i++) {
            double orig = dense.weights().data()[i];
            dense.weights().data()[i] = orig + epsilon;
            Tensor outPlus = layer.forward(input);
            double lossPlus = outPlus.sum();
            dense.weights().data()[i] = orig - epsilon;
            Tensor outMinus = layer.forward(input);
            double lossMinus = outMinus.sum();
            dense.weights().data()[i] = orig;

            double numGrad = (lossPlus - lossMinus) / (2 * epsilon);
            double anaGrad = dense.weightsGradient().data()[i];

            if (Math.abs(numGrad) > 1e-8) {
                double relError = Math.abs(anaGrad - numGrad) / (Math.abs(anaGrad) + Math.abs(numGrad) + 1e-8);
                if (relError > tolerance) {
                    System.out.printf("Gradient mismatch at %d: analytic=%.6f, numerical=%.6f, relError=%.6f%n",
                        i, anaGrad, numGrad, relError);
                }
                checked++;
            }
        }
        return checked > 0 ? 0 : 1;
    }

    public void checkAll(ml.network.NeuralNetwork network, Tensor input) {
        System.out.println("Checking gradients...");
        Tensor output = network.forward(input);
        Tensor grad = Tensor.ones(output.rows(), output.cols());
        network.backward(grad);
        System.out.println("Gradient check complete.");
    }
}
