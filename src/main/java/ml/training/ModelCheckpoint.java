package ml.training;

import ml.network.NeuralNetwork;
import ml.network.ModelSerializer;

import java.io.File;

public class ModelCheckpoint implements Callback {
    private final String filepath;
    private final String monitor;
    private final boolean saveBestOnly;
    private final boolean verbose;
    private double bestValue;
    private String bestMode;

    public ModelCheckpoint(String filepath, String monitor, boolean saveBestOnly, boolean verbose) {
        this.filepath = filepath;
        this.monitor = monitor.toLowerCase();
        this.saveBestOnly = saveBestOnly;
        this.verbose = verbose;
        this.bestValue = Double.POSITIVE_INFINITY;
        this.bestMode = "min";
    }

    public ModelCheckpoint(String filepath, String monitor) {
        this(filepath, monitor, true, true);
    }

    public ModelCheckpoint(String filepath) {
        this(filepath, "loss", true, true);
    }

    @Override
    public void onEpochEnd(int epoch, double loss, double valLoss) {
        double value = monitor.equals("val_loss") ? valLoss : loss;
        boolean improved = bestMode.equals("min") ? value < bestValue : value > bestValue;

        if (improved || !saveBestOnly) {
            if (improved) bestValue = value;

            String path = filepath.replace("{epoch}", String.valueOf(epoch))
                                  .replace("{loss}", String.format("%.4f", loss));

            try {
                NeuralNetwork tempNet = new NeuralNetwork();
                ModelSerializer.saveBinary(tempNet, path);
                if (verbose) {
                    System.out.printf("ModelCheckpoint: Saved model to %s (monitor=%s, value=%.4f)%n",
                        path, monitor, value);
                }
            } catch (Exception e) {
                System.err.println("ModelCheckpoint: Failed to save model - " + e.getMessage());
            }
        }
    }

    public void saveModel(NeuralNetwork network, int epoch, double loss) {
        String path = filepath.replace("{epoch}", String.valueOf(epoch))
                              .replace("{loss}", String.format("%.4f", loss));
        try {
            ModelSerializer.saveBinary(network, path);
            if (verbose) {
                System.out.printf("ModelCheckpoint: Saved model to %s%n", path);
            }
        } catch (Exception e) {
            System.err.println("ModelCheckpoint: Failed to save model - " + e.getMessage());
        }
    }

    public String getFilepath() { return filepath; }
    public String getMonitor() { return monitor; }
    public boolean isSaveBestOnly() { return saveBestOnly; }
    public double getBestValue() { return bestValue; }
}
