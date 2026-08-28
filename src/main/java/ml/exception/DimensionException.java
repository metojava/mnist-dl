package ml.exception;

public class DimensionException extends RuntimeException {
    public DimensionException(String message) {
        super(message);
    }

    public static DimensionException unexpectedRank(String operation, int expected, int actual) {
        return new DimensionException(String.format(
            "%s: expected rank %d but got %d", operation, expected, actual));
    }
}
