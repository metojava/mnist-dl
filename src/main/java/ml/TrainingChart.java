package ml;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TrainingChart {

    private final List<Integer> epochs = new ArrayList<>();
    private final List<Double> losses = new ArrayList<>();
    private final List<Double> accuracies = new ArrayList<>();

    public void addEpoch(
            int epoch,
            double loss,
            double accuracy
    ) {
        epochs.add(epoch);
        losses.add(loss);
        accuracies.add(accuracy);
    }

    public void save(String outputFile)
            throws IOException {

        DefaultCategoryDataset dataset =
                new DefaultCategoryDataset();

        for (int i = 0; i < epochs.size(); i++) {

            dataset.addValue(
                    losses.get(i),
                    "Loss",
                    String.valueOf(epochs.get(i))
            );

            dataset.addValue(
                    accuracies.get(i),
                    "Accuracy",
                    String.valueOf(epochs.get(i))
            );
        }

        JFreeChart chart =
                ChartFactory.createLineChart(
                        "MNIST Training",
                        "Epoch",
                        "Value",
                        dataset,
                        PlotOrientation.VERTICAL,
                        true,
                        true,
                        false
                );

        ChartUtils.saveChartAsPNG(
                new File(outputFile),
                chart,
                1000,
                600
        );
    }
}
