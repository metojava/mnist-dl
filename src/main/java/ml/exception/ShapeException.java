package ml.exception;

public class ShapeException extends RuntimeException {
    public ShapeException(String message) {
        super(message);
    }

    public ShapeException(String message, Throwable cause) {
        super(message, cause);
    }

    public static ShapeException mismatch(String operation, int[] expected, int[] actual) {
        return new ShapeException(String.format(
            "%s: expected shape %s but got %s",
            operation, java.util.Arrays.toString(expected), java.util.Arrays.toString(actual)));
    }

    public static ShapeException incompatible(String operation, String detail) {
        return new ShapeException(String.format("%s: %s", operation, detail));
    }
}
