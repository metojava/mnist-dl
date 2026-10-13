package ml;

import java.util.Random;

public class Trainer {

    private final NeuralNetwork network;
    private final SGD optimizer;
    private final Loss lossFunction;

    private final Random random = new Random();

    private final TrainingChart trainingChart =
            new TrainingChart();

    public Trainer(
            NeuralNetwork network,
            SGD optimizer,
            Loss lossFunction
    ) {
        this.network = network;
        this.optimizer = optimizer;
        this.lossFunction = lossFunction;
    }

    public void fit(
            Tensor x,
            int[] y,
            int epochs,
            int batchSize
    ) {

        if (x.rows() != y.length) {
            throw new IllegalArgumentException(
                    "Number of examples and labels must match."
            );
        }

        int numberOfSamples = x.rows();

        int imageColumns = x.cols();

        float[] imageData =
                x.data();

        int[] indices =
                new int[numberOfSamples];

        for (int i = 0;
             i < numberOfSamples;
             i++) {

            indices[i] = i;
        }

        /*
         * Reusable batch buffers. Garbage avoided for the common
         * full-size batches; only a fractional final batch allocates.
         */
        Tensor batchX = null;
        int[] batchY = null;

        long totalTrainingStart =
                System.nanoTime();

        for (int epoch = 1;
             epoch <= epochs;
             epoch++) {

            long epochStart =
                    System.nanoTime();

            shuffle(indices);

            double epochLoss = 0.0;
            int batches = 0;

            // =====================================================
            // Mini-batch training
            // =====================================================

            for (int start = 0;
                 start < numberOfSamples;
                 start += batchSize) {

                int end =
                        Math.min(
                                start + batchSize,
                                numberOfSamples
                        );

                int currentBatchSize =
                        end - start;

                /*
                 * Reuse the batch buffers when the batch is full;
                 * allocate only for the (usually smaller) last batch.
                 */
                if (currentBatchSize == batchSize) {

                    if (batchX == null ||
                        batchX.rows() != batchSize) {

                        batchX =
                                new Tensor(
                                        batchSize,
                                        imageColumns
                                );

                        batchY =
                                new int[batchSize];
                    }

                } else {

                    batchX =
                            new Tensor(
                                    currentBatchSize,
                                    imageColumns
                            );

                    batchY =
                            new int[currentBatchSize];
                }

                /*
                 * Gather the batch rows with a single arraycopy
                 * each instead of per-element reads.
                 */
                float[] batchData =
                        batchX.data();

                for (int i = 0;
                     i < currentBatchSize;
                     i++) {

                    int index =
                            indices[start + i];

                    System.arraycopy(
                            imageData,
                            index * imageColumns,
                            batchData,
                            i * imageColumns,
                            imageColumns
                    );

                    batchY[i] =
                            y[index];
                }

                // Forward
                Tensor predictions =
                        network.forward(batchX);

                // Loss
                double loss =
                        lossFunction.forward(
                                predictions,
                                batchY
                        );

                epochLoss += loss;

                batches++;

                // Backpropagation
                Tensor gradient =
                        lossFunction.backward(
                                predictions,
                                batchY
                        );

                network.backward(gradient);

                // Update weights
                network.update(
                        optimizer.getLearningRate()
                );
            }

            double averageLoss =
                    epochLoss / batches;

            long epochEnd =
                    System.nanoTime();

            double epochSeconds =
                    (epochEnd - epochStart)
                            / 1_000_000_000.0;

            // Save history.
            // Accuracy is not calculated here because
            // evaluating all 60,000 training samples after
            // every epoch is expensive.
            trainingChart.addEpoch(
                    epoch,
                    averageLoss,
                    0.0
            );

            System.out.printf(
                    "Epoch %d/%d - " +
                    "loss: %.4f - " +
                    "time: %.2f sec%n",

                    epoch,
                    epochs,
                    averageLoss,
                    epochSeconds
            );
        }

        long totalTrainingEnd =
                System.nanoTime();

        double totalTrainingSeconds =
                (totalTrainingEnd - totalTrainingStart)
                        / 1_000_000_000.0;

        System.out.println();

        System.out.printf(
                "Total training time: %.2f sec%n",
                totalTrainingSeconds
        );

        System.out.printf(
                "Average epoch time: %.2f sec%n",
                totalTrainingSeconds / epochs
        );
    }

    public void saveTrainingChart(
            String outputFile
    ) throws Exception {

        trainingChart.save(
                outputFile
        );
    }

    private void shuffle(int[] array) {

        for (int i = array.length - 1;
             i > 0;
             i--) {

            int j =
                    random.nextInt(i + 1);

            int temp =
                    array[i];

            array[i] =
                    array[j];

            array[j] =
                    temp;
        }
    }
}
