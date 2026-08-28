package ml.training;

import ml.network.NeuralNetwork;

public interface Callback {
    default void onTrainBegin(int totalEpochs) {}
    default void onTrainEnd() {}
    default void onEpochBegin(int epoch) {}
    default void onEpochEnd(int epoch, double loss, double valLoss) {}
    default void onBatchBegin(int batch) {}
    default void onBatchEnd(int batch, double batchLoss) {}
    default void onLrChange(double newLr) {}

    static Callback chain(Callback... callbacks) {
        return new Callback() {
            @Override
            public void onTrainBegin(int totalEpochs) {
                for (Callback c : callbacks) c.onTrainBegin(totalEpochs);
            }
            @Override
            public void onTrainEnd() {
                for (Callback c : callbacks) c.onTrainEnd();
            }
            @Override
            public void onEpochBegin(int epoch) {
                for (Callback c : callbacks) c.onEpochBegin(epoch);
            }
            @Override
            public void onEpochEnd(int epoch, double loss, double valLoss) {
                for (Callback c : callbacks) c.onEpochEnd(epoch, loss, valLoss);
            }
            @Override
            public void onBatchBegin(int batch) {
                for (Callback c : callbacks) c.onBatchBegin(batch);
            }
            @Override
            public void onBatchEnd(int batch, double batchLoss) {
                for (Callback c : callbacks) c.onBatchEnd(batch, batchLoss);
            }
            @Override
            public void onLrChange(double newLr) {
                for (Callback c : callbacks) c.onLrChange(newLr);
            }
        };
    }
}
