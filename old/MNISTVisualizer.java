package ml;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYTextAnnotation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.data.xy.DefaultXYZDataset;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTVisualizer {

    private static final int IMAGE_SIZE = 28;

    private static final int CELL_SIZE = 140;

    private static final int LABEL_HEIGHT = 85;

    private static final int GAP = 20;

    private static final Color CORRECT_COLOR =
            new Color(45, 180, 100);

    private static final Color WRONG_COLOR =
            new Color(220, 70, 80);

    private static final Color BACKGROUND =
            new Color(245, 247, 250);

    private static final Color TEXT =
            new Color(35, 40, 50);

    private MNISTVisualizer() {
    }

    /**
     * Creates a PNG containing MNIST images,
     * predictions, confidence and actual labels.
     *
     * JFreeChart is used as the rendering/charting
     * component while Java2D is used for the
     * image cells and textual annotations.
     */
    public static void savePredictions(
            NeuralNetwork network,
            Tensor images,
            int[] labels,
            int startIndex,
            int numberOfImages,
            String outputFile
    ) throws IOException {

        if (startIndex < 0 ||
                startIndex >= images.rows()) {

            throw new IllegalArgumentException(
                    "Invalid start index."
            );
        }

        int count =
                Math.min(
                        numberOfImages,
                        images.rows() - startIndex
                );

        PredictionResult result =
                new Predictor(network)
                        .predict(images, labels);

        int columns = 5;

        int rows =
                (int) Math.ceil(
                        (double) count / columns
                );

        int width =
                columns * (CELL_SIZE + GAP) + GAP;

        int height =
                rows * (CELL_SIZE + LABEL_HEIGHT + GAP)
                + GAP
                + 60;

        BufferedImage canvas =
                new BufferedImage(
                        width,
                        height,
                        BufferedImage.TYPE_INT_RGB
                );

        Graphics2D graphics =
                canvas.createGraphics();

        graphics.setRenderingHint(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON
        );

        graphics.setRenderingHint(
                RenderingHints.KEY_TEXT_ANTIALIASING,
                RenderingHints.VALUE_TEXT_ANTIALIAS_ON
        );

        graphics.setColor(BACKGROUND);

        graphics.fillRect(
                0,
                0,
                width,
                height
        );

        Prediction[] predictions =
                result.predictions();

        for (int i = 0; i < count; i++) {

            int datasetIndex =
                    startIndex + i;

            int column =
                    i % columns;

            int row =
                    i / columns;

            int x =
                    GAP + column *
                    (CELL_SIZE + GAP);

            int y =
                    50 + row *
                    (CELL_SIZE + LABEL_HEIGHT + GAP);

            Prediction prediction =
                    predictions[datasetIndex];

            drawImage(
                    graphics,
                    images,
                    datasetIndex,
                    x,
                    y
            );

            drawPredictionText(
                    graphics,
                    prediction,
                    x,
                    y + CELL_SIZE + 8
            );
        }

        graphics.setFont(
                new Font(
                        "SansSerif",
                        Font.BOLD,
                        18
                )
        );

        graphics.setColor(TEXT);

        String title =
                String.format(
                        "MNIST Predictions — Accuracy: %.2f%%",
                        result.accuracyPercentage()
                );

        graphics.drawString(
                title,
                GAP,
                30
        );

        graphics.dispose();

        ImageIO.write(
                canvas,
                "png",
                new File(outputFile)
        );
    }

    private static void drawImage(
            Graphics2D graphics,
            Tensor images,
            int imageIndex,
            int x,
            int y
    ) {

        BufferedImage image =
                new BufferedImage(
                        CELL_SIZE,
                        CELL_SIZE,
                        BufferedImage.TYPE_INT_RGB
                );

        Graphics2D imageGraphics =
                image.createGraphics();

        imageGraphics.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR
        );

        imageGraphics.setRenderingHint(
                RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_SPEED
        );

        for (int row = 0; row < IMAGE_SIZE; row++) {

            for (int col = 0; col < IMAGE_SIZE; col++) {

                double value =
                        images.get(
                                imageIndex,
                                row * IMAGE_SIZE + col
                        );

                value =
                        Math.max(
                                0.0,
                                Math.min(
                                        1.0,
                                        value
                                )
                        );

                int gray =
                        (int) Math.round(
                                value * 255
                        );

                imageGraphics.setColor(
                        new Color(
                                gray,
                                gray,
                                gray
                        )
                );

                double pixelWidth =
                        (double) CELL_SIZE /
                        IMAGE_SIZE;

                double pixelHeight =
                        (double) CELL_SIZE /
                        IMAGE_SIZE;

                imageGraphics.fill(
                        new Rectangle2D.Double(
                                col * pixelWidth,
                                row * pixelHeight,
                                pixelWidth + 1,
                                pixelHeight + 1
                        )
                );
            }
        }

        imageGraphics.dispose();

        graphics.drawImage(
                image,
                x,
                y,
                null
        );

        graphics.setColor(
                Color.DARK_GRAY
        );

        graphics.setStroke(
                new BasicStroke(1.5f)
        );

        graphics.drawRect(
                x,
                y,
                CELL_SIZE,
                CELL_SIZE
        );
    }

    private static void drawPredictionText(
            Graphics2D graphics,
            Prediction prediction,
            int x,
            int y
    ) {

        Color resultColor =
                prediction.correct()
                        ? CORRECT_COLOR
                        : WRONG_COLOR;

        graphics.setFont(
                new Font(
                        "SansSerif",
                        Font.BOLD,
                        14
                )
        );

        graphics.setColor(resultColor);

        String predictionText =
                String.format(
                        "Predicted: %d",
                        prediction.predictedLabel()
                );

        graphics.drawString(
                predictionText,
                x,
                y
        );

        graphics.setFont(
                new Font(
                        "SansSerif",
                        Font.PLAIN,
                        12
                )
        );

        graphics.setColor(TEXT);

        String confidenceText =
                String.format(
                        "Confidence: %.2f%%",
                        prediction.confidencePercentage()
                );

        graphics.drawString(
                confidenceText,
                x,
                y + 18
        );

        String actualText =
                String.format(
                        "Actual: %d",
                        prediction.actualLabel()
                );

        graphics.drawString(
                actualText,
                x,
                y + 36
        );

        String status =
                prediction.correct()
                        ? "✓ Correct"
                        : "✗ Wrong";

        graphics.setColor(resultColor);

        graphics.setFont(
                new Font(
                        "SansSerif",
                        Font.BOLD,
                        12
                )
        );

        graphics.drawString(
                status,
                x,
                y + 54
        );
    }
}
