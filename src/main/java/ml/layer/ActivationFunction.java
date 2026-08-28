package ml.layer;

public interface ActivationFunction {
    double forward(double x);
    double derivative(double x);
}
