package ml.training;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class TensorBoardLogger implements Callback {
    private final String logDir;
    private final String runName;
    private final boolean verbose;
    private final Map<String, List<ScalarEntry>> scalarBuffers;
    private final AtomicInteger globalStep;
    private final List<HistogramEntry> histogramBuffer;

    public TensorBoardLogger(String logDir, String runName, boolean verbose) {
        this.logDir = logDir;
        this.runName = runName;
        this.verbose = verbose;
        this.scalarBuffers = new ConcurrentHashMap<>();
        this.globalStep = new AtomicInteger(0);
        this.histogramBuffer = Collections.synchronizedList(new ArrayList<>());

        new File(logDir).mkdirs();
    }

    public TensorBoardLogger(String logDir, String runName) {
        this(logDir, runName, true);
    }

    public TensorBoardLogger(String logDir) {
        this(logDir, "run_" + System.currentTimeMillis());
    }

    public void logScalar(String tag, double value, int step) {
        scalarBuffers.computeIfAbsent(tag, k -> new ArrayList<>())
                     .add(new ScalarEntry(step, value));
    }

    public void logScalars(Map<String, Double> scalars, int step) {
        for (var entry : scalars.entrySet()) {
            logScalar(entry.getKey(), entry.getValue(), step);
        }
    }

    public void logHistogram(String tag, double[] values, int step) {
        histogramBuffer.add(new HistogramEntry(tag, step, values));
    }

    @Override
    public void onTrainBegin(int totalEpochs) {
        if (verbose) {
            System.out.printf("TensorBoardLogger: Logging to %s (run: %s)%n", logDir, runName);
        }
    }

    @Override
    public void onEpochEnd(int epoch, double loss, double valLoss) {
        int step = globalStep.incrementAndGet();
        logScalar("loss", loss, step);
        if (valLoss >= 0) {
            logScalar("val_loss", valLoss, step);
        }
    }

    @Override
    public void onLrChange(double newLr) {
        int step = globalStep.get();
        logScalar("learning_rate", newLr, step);
    }

    @Override
    public void onTrainEnd() {
        flush();
    }

    public void flush() {
        try {
            String eventFile = logDir + File.separator + "events.out.tfevents." +
                             System.currentTimeMillis() + "." + runName;
            DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(eventFile)));

            for (var entry : scalarBuffers.entrySet()) {
                String tag = entry.getKey();
                for (ScalarEntry se : entry.getValue()) {
                    writeEvent(out, tag, se.step(), se.value());
                }
            }

            for (HistogramEntry he : histogramBuffer) {
                writeHistogram(out, he.tag(), he.step(), he.values());
            }

            out.flush();
            out.close();

            if (verbose) {
                System.out.printf("TensorBoardLogger: Flushed %d scalars, %d histograms to %s%n",
                    scalarBuffers.values().stream().mapToInt(List::size).sum(),
                    histogramBuffer.size(), eventFile);
            }
        } catch (IOException e) {
            System.err.println("TensorBoardLogger: Failed to flush - " + e.getMessage());
        }
    }

    private void writeEvent(DataOutputStream out, String tag, int step, double value) throws IOException {
        writeString(out, tag);
        writeDouble(out, value);
        writeVarint(out, step);
        writeVarint(out, 0);
    }

    private void writeHistogram(DataOutputStream out, String tag, int step, double[] values) throws IOException {
        writeString(out, tag);
        writeDouble(out, 0);
        writeVarint(out, step);
        writeVarint(out, 2);

        Arrays.sort(values);
        int n = values.length;
        double min = n > 0 ? values[0] : 0;
        double max = n > 0 ? values[n - 1] : 0;
        double mean = n > 0 ? Arrays.stream(values).average().orElse(0) : 0;
        double variance = 0;
        for (double v : values) variance += (v - mean) * (v - mean);
        variance = n > 0 ? variance / n : 0;

        writeVarint(out, n);
        writeDouble(out, min);
        writeDouble(out, max);
        writeDouble(out, mean);
        writeDouble(out, variance);

        int numBuckets = Math.min(30, Math.max(1, (int) Math.ceil(Math.sqrt(n))));
        double bucketWidth = (max - min) / numBuckets;
        if (bucketWidth <= 0) bucketWidth = 1;

        for (int i = 0; i < numBuckets; i++) {
            double lower = min + i * bucketWidth;
            double upper = lower + bucketWidth;
            int count = 0;
            for (double v : values) {
                if (v >= lower && (i == numBuckets - 1 ? v <= upper : v < upper)) count++;
            }
            writeDouble(out, lower);
            writeVarint(out, count);
        }
    }

    private void writeString(DataOutputStream out, String s) throws IOException {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        writeVarint(out, bytes.length);
        out.write(bytes);
    }

    private void writeDouble(DataOutputStream out, double value) throws IOException {
        out.writeLong(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(value).getLong());
    }

    private void writeVarint(DataOutputStream out, int value) throws IOException {
        while ((value & ~0x7F) != 0) {
            out.writeByte((value & 0x7F) | 0x80);
            value >>>= 7;
        }
        out.writeByte(value);
    }

    public Map<String, List<ScalarEntry>> getScalarBuffers() { return Collections.unmodifiableMap(scalarBuffers); }
    public int getStep() { return globalStep.get(); }

    private record ScalarEntry(int step, double value) {}
    private record HistogramEntry(String tag, int step, double[] values) {}
}
