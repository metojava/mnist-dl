package ml.util;

import ml.core.Tensor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTVisualizer {
    private static final String[] FASHION_LABELS = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    private final String[] classNames;

    public MNISTVisualizer(String[] classNames) {
        this.classNames = classNames;
    }

    public MNISTVisualizer() {
        this(FASHION_LABELS);
    }

    public void visualize(Tensor images, int[] predictedLabels, int[] actualLabels,
                          double[] confidences, int numSamples, String outputFile) {
        int cols = 4;
        int rowsPerCard = 1;
        int cardsPerRow = cols;
        int totalRows = (int) Math.ceil((double) numSamples / cardsPerRow);
        int cellW = 280;
        int cellH = 120;
        int imgSize = 100;
        int padding = 20;

        int imgW = cardsPerRow * cellW + padding * 2;
        int imgH = totalRows * cellH + padding * 2;
        BufferedImage image = new BufferedImage(imgW, imgH, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = image.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, imgW, imgH);

        for (int idx = 0; idx < numSamples && idx < images.rows(); idx++) {
            int row = idx / cardsPerRow;
            int col = idx % cardsPerRow;
            int x = padding + col * cellW;
            int y = padding + row * cellH;

            BufferedImage digit = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for (int py = 0; py < 28; py++)
                for (int px = 0; px < 28; px++) {
                    int val = (int) (images.get(idx, py * 28 + px) * 255);
                    digit.getRaster().setSample(px, py, 0, val);
                }
            g.drawImage(digit, x, y, imgSize, imgSize, null);

            boolean correct = predictedLabels[idx] == actualLabels[idx];
            String predName = classNames != null ? classNames[predictedLabels[idx]] : String.valueOf(predictedLabels[idx]);
            String actName = classNames != null ? classNames[actualLabels[idx]] : String.valueOf(actualLabels[idx]);

            g.setFont(new Font("Arial", Font.BOLD, 11));
            g.setColor(correct ? new Color(0, 128, 0) : new Color(200, 0, 0));
            g.drawString(predName + " (" + String.format("%.0f", confidences[idx] * 100) + "%)",
                x + imgSize + 5, y + 15);

            g.setFont(new Font("Arial", Font.PLAIN, 10));
            g.setColor(Color.DARK_GRAY);
            g.drawString("Actual: " + actName, x + imgSize + 5, y + 35);

            g.setColor(correct ? new Color(0, 150, 0) : new Color(220, 0, 0));
            g.drawString(correct ? "CORRECT" : "WRONG", x + imgSize + 5, y + 55);
        }

        g.dispose();
        try {
            ImageIO.write(image, "png", new File(outputFile));
            System.out.println("Predictions saved to " + outputFile);
        } catch (IOException e) {
            System.err.println("Failed to save visualization: " + e.getMessage());
        }
    }
}
