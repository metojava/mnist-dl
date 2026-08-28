package ml.loss;

import ml.core.Tensor;

public interface Loss {
    double forward(Tensor predictions, int[] labels);
    Tensor backward(Tensor predictions, int[] labels);
}
