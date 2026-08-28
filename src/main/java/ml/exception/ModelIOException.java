package ml.exception;

public class ModelIOException extends RuntimeException {
    public ModelIOException(String message) {
        super(message);
    }

    public ModelIOException(String message, Throwable cause) {
        super(message, cause);
    }
}
