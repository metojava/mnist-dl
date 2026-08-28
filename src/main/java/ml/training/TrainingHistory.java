package ml.training;

import java.util.ArrayList;
import java.util.List;

public class TrainingHistory {
    private final List<EpochRecord> records;

    public record EpochRecord(int epoch, double loss, long timeMs) {}

    public TrainingHistory(List<EpochRecord> records) {
        this.records = new ArrayList<>(records);
    }

    public List<EpochRecord> records() { return records; }
    public int totalEpochs() { return records.size(); }

    public double lastLoss() {
        return records.isEmpty() ? 0 : records.get(records.size() - 1).loss();
    }

    public double bestLoss() {
        return records.stream().mapToDouble(EpochRecord::loss).min().orElse(0);
    }

    public long totalTimeMs() {
        return records.stream().mapToLong(EpochRecord::timeMs).sum();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("TrainingHistory[\n");
        for (var r : records)
            sb.append(String.format("  Epoch %d: loss=%.6f, time=%dms%n", r.epoch(), r.loss(), r.timeMs()));
        sb.append("]");
        return sb.toString();
    }
}
