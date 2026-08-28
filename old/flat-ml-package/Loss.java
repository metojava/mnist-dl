package ml;

public interface Loss {

    double forward(Tensor predictions, int[] labels);

    Tensor backward(Tensor predictions, int[] labels);
}
