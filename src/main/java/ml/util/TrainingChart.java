package ml.util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TrainingChart {
    private final List<Integer> epochs = new ArrayList<>();
    private final List<Double> losses = new ArrayList<>();
    private final List<Double> accuracies = new ArrayList<>();

    public void addEpoch(int epoch, double loss) {
        epochs.add(epoch);
        losses.add(loss);
    }

    public void addEpoch(int epoch, double loss, double accuracy) {
        addEpoch(epoch, loss);
        accuracies.add(accuracy);
    }

    public void save(String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < epochs.size(); i++) {
            dataset.addValue(losses.get(i), "Loss", epochs.get(i));
        }
        if (!accuracies.isEmpty()) {
            for (int i = 0; i < epochs.size(); i++)
                dataset.addValue(accuracies.get(i), "Accuracy", epochs.get(i));
        }

        JFreeChart chart = ChartFactory.createLineChart(
            "Training History", "Epoch", "Value", dataset);
        chart.setBackgroundPaint(Color.WHITE);

        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.WHITE);
        LineAndShapeRenderer renderer = (LineAndShapeRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(0, 120, 215));
        if (!accuracies.isEmpty())
            renderer.setSeriesPaint(1, new Color(0, 180, 80));

        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, 1000, 600);
            System.out.println("Chart saved to " + outputFile);
        } catch (IOException e) {
            System.err.println("Failed to save chart: " + e.getMessage());
        }
    }
}
