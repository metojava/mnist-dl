package ml.training;

import ml.core.Tensor;

public record Dataset(Tensor images, int[] labels) {
    public int size() { return images.rows(); }
    public int features() { return images.cols(); }
    public int classes() {
        int max = 0;
        for (int l : labels) if (l > max) max = l;
        return max + 1;
    }

    public Dataset split(int trainSize) {
        Tensor trainImages = images.sliceRows(0, trainSize);
        int[] trainLabels = new int[trainSize];
        System.arraycopy(labels, 0, trainLabels, 0, trainSize);
        Tensor testImages = images.sliceRows(trainSize, images.rows());
        int[] testLabels = new int[labels.length - trainSize];
        System.arraycopy(labels, trainSize, testLabels, 0, testLabels.length);
        return new Dataset(trainImages, trainLabels);
    }
}
