package projects.tabular;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import projects.common.Proj;

/**
 * P12 - Iris flower classification. The complete classic Fisher/Anderson dataset
 * (150 samples, 4 features, 3 species) is embedded below so the project runs offline.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.tabular.IrisClassifier" "-Dexec.args=120 16"
 */
public class IrisClassifier {
    // sepal length, sepal width, petal length, petal width (cm)
    static final double[][] DATA = {
        {5.1,3.5,1.4,0.2},{4.9,3.0,1.4,0.2},{4.7,3.2,1.3,0.2},{4.6,3.1,1.5,0.2},{5.0,3.6,1.4,0.2},
        {5.4,3.9,1.7,0.4},{4.6,3.4,1.4,0.3},{5.0,3.4,1.5,0.2},{4.4,2.9,1.4,0.2},{4.9,3.1,1.5,0.1},
        {5.4,3.7,1.5,0.2},{4.8,3.4,1.6,0.2},{4.8,3.0,1.4,0.1},{4.3,3.0,1.1,0.1},{5.8,4.0,1.2,0.2},
        {5.7,4.4,1.5,0.4},{5.4,3.9,1.3,0.4},{5.1,3.5,1.4,0.3},{5.7,3.8,1.7,0.3},{5.1,3.8,1.5,0.3},
        {5.4,3.4,1.7,0.2},{5.1,3.7,1.5,0.4},{4.6,3.6,1.0,0.2},{5.1,3.3,1.7,0.5},{4.8,3.4,1.9,0.2},
        {5.0,3.0,1.6,0.2},{5.0,3.4,1.6,0.4},{5.2,3.5,1.5,0.2},{5.2,3.4,1.4,0.2},{4.7,3.2,1.6,0.2},
        {4.8,3.1,1.6,0.2},{5.4,3.4,1.5,0.4},{5.2,4.1,1.5,0.1},{5.5,4.2,1.4,0.2},{4.9,3.1,1.5,0.2},
        {5.0,3.2,1.2,0.2},{5.5,3.5,1.3,0.2},{4.9,3.6,1.4,0.1},{4.4,3.0,1.3,0.2},{5.1,3.4,1.5,0.2},
        {5.0,3.5,1.3,0.3},{4.5,2.3,1.3,0.3},{4.4,3.2,1.3,0.2},{5.0,3.5,1.6,0.6},{5.1,3.8,1.9,0.4},
        {4.8,3.0,1.4,0.3},{5.1,3.8,1.6,0.2},{4.6,3.2,1.4,0.2},{5.3,3.7,1.5,0.2},{5.0,3.3,1.4,0.2},
        {7.0,3.2,4.7,1.4},{6.4,3.2,4.5,1.5},{6.9,3.1,4.9,1.5},{5.5,2.3,4.0,1.3},{6.5,2.8,4.6,1.5},
        {5.7,2.8,4.5,1.3},{6.3,3.3,4.7,1.6},{4.9,2.4,3.3,1.0},{6.6,2.9,4.6,1.3},{5.2,2.7,3.9,1.4},
        {5.0,2.0,3.5,1.0},{5.9,3.0,4.2,1.5},{6.0,2.2,4.0,1.0},{6.1,2.9,4.7,1.4},{5.6,2.9,3.6,1.3},
        {6.7,3.1,4.4,1.4},{5.6,3.0,4.5,1.5},{5.8,2.7,4.1,1.0},{6.2,2.2,4.5,1.5},{5.6,2.5,3.9,1.1},
        {5.9,3.2,4.8,1.8},{6.1,2.8,4.0,1.3},{6.3,2.5,4.9,1.5},{6.1,2.8,4.7,1.2},{6.4,2.9,4.3,1.3},
        {6.6,3.0,4.4,1.4},{6.8,2.8,4.8,1.4},{6.7,3.0,5.0,1.7},{6.0,2.9,4.5,1.5},{5.7,2.6,3.5,1.0},
        {5.5,2.4,3.8,1.1},{5.5,2.4,3.7,1.0},{5.8,2.7,3.9,1.2},{6.0,2.7,5.1,1.6},{5.4,3.0,4.5,1.5},
        {6.0,3.4,4.5,1.6},{6.7,3.1,4.7,1.5},{6.3,2.3,4.4,1.3},{5.6,3.0,4.1,1.3},{5.5,2.5,4.0,1.3},
        {5.5,2.6,4.4,1.2},{6.1,3.0,4.6,1.4},{5.8,2.6,4.0,1.2},{5.0,2.3,3.3,1.0},{5.6,2.7,4.2,1.3},
        {5.7,3.0,4.2,1.2},{5.7,2.9,4.2,1.3},{6.2,2.9,4.3,1.3},{5.1,2.5,3.0,1.1},{5.7,2.8,4.1,1.3},
        {6.3,3.3,6.0,2.5},{5.8,2.7,5.1,1.9},{7.1,3.0,5.9,2.1},{6.3,2.9,5.6,1.8},{6.5,3.0,5.8,2.2},
        {7.6,3.0,6.6,2.1},{4.9,2.5,4.5,1.7},{7.3,2.9,6.3,1.8},{6.7,2.5,5.8,1.8},{7.2,3.6,6.1,2.5},
        {6.5,3.2,5.1,2.0},{6.4,2.7,5.3,1.9},{6.8,3.0,5.5,2.1},{5.7,2.5,5.0,2.0},{5.8,2.8,5.1,2.4},
        {6.4,3.2,5.3,2.3},{6.5,3.0,5.5,1.8},{7.7,3.8,6.7,2.2},{7.7,2.6,6.9,2.3},{6.0,2.2,5.0,1.5},
        {6.9,3.2,5.7,2.3},{5.6,2.8,4.9,2.0},{7.7,2.8,6.7,2.0},{6.3,2.7,4.9,1.8},{6.7,3.3,5.7,2.1},
        {7.2,3.2,6.0,1.8},{6.2,2.8,4.8,1.8},{6.1,3.0,4.9,1.8},{6.4,2.8,5.6,2.1},{7.2,3.0,5.8,1.6},
        {7.4,2.8,6.1,1.9},{7.9,3.8,6.4,2.0},{6.4,2.8,5.6,2.2},{6.3,2.8,5.1,1.5},{6.1,2.6,5.6,1.4},
        {7.7,3.0,6.1,2.3},{6.3,3.4,5.6,2.4},{6.4,3.1,5.5,1.8},{6.0,3.0,4.8,1.8},{6.9,3.1,5.4,2.1},
        {6.7,3.1,5.6,2.4},{6.9,3.1,5.1,2.3},{5.8,2.7,5.1,1.9},{6.8,3.2,5.9,2.3},{6.7,3.3,5.7,2.5},
        {6.7,3.0,5.2,2.3},{6.3,2.5,5.0,1.9},{6.5,3.0,5.2,2.0},{6.2,3.4,5.4,2.3},{5.9,3.0,5.1,1.8}};

    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 120), batch = Proj.batch(args, 16);
        long t0 = System.currentTimeMillis();

        int n = DATA.length;
        Tensor x = new Tensor(n, 4);
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 4; j++) x.set(i, j, DATA[i][j]);
            y[i] = i / 50;   // 50 rows per species
        }
        Proj.standardize(x);
        Proj.shuffle(x, y, new java.util.Random(42));

        int split = (int) (n * 0.75);
        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(4, 16));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(16, 3));
        net.add(new ml.activation.Softmax());
        System.out.printf("[P12 Iris] samples=%d split=%d%n", n, split);

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.01))
            .fit(x.sliceRows(0, split), java.util.Arrays.copyOf(y, split), epochs, batch);

        String[] names = {"setosa", "versicolor", "virginica"};
        System.out.printf("Train acc: %.1f%% | Test acc: %.1f%%%n",
            100 * acc(net, x.sliceRows(0, split), java.util.Arrays.copyOf(y, split)),
            100 * acc(net, x.sliceRows(split, n), java.util.Arrays.copyOfRange(y, split, n)));

        double[][] demo = {{5.1, 3.5, 1.4, 0.2}, {6.0, 2.9, 4.5, 1.5}, {6.9, 3.1, 5.4, 2.1}};
        for (double[] d : demo) {
            Tensor row = Tensor.fromFlat(standardizeRow(d), 1, 4);
            Tensor out = net.forward(row, false);
            System.out.printf("  %s -> %s (%.0f%%)%n", java.util.Arrays.toString(d),
                names[out.argmaxRow(0)], 100 * out.data()[out.argmaxRow(0)]);
        }
        Proj.footer("P12 Iris", t0);
    }

    private static double acc(NeuralNetwork net, Tensor x, int[] y) {
        Tensor out = net.forward(x, false);
        int c = 0;
        for (int i = 0; i < x.rows(); i++) if (out.argmaxRow(i) == y[i]) c++;
        return (double) c / x.rows();
    }

    private static double[][] stats() {
        double[] mean = new double[4], std = new double[4];
        for (int j = 0; j < 4; j++) {
            double s = 0;
            for (double[] r : DATA) s += r[j];
            mean[j] = s / DATA.length;
            s = 0;
            for (double[] r : DATA) { double d = r[j] - mean[j]; s += d * d; }
            std[j] = Math.sqrt(s / (DATA.length - 1));
        }
        return new double[][]{mean, std};
    }

    private static double[] standardizeRow(double[] row) {
        double[][] ms = stats();
        double[] out = new double[4];
        for (int j = 0; j < 4; j++) out[j] = (row[j] - ms[0][j]) / ms[1][j];
        return out;
    }
}
