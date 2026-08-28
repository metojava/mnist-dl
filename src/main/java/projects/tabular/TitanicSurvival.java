package projects.tabular;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import projects.common.Proj;
import java.util.List;
import java.util.ArrayList;
import ml.regularization.Dropout;

/**
 * P16 - Titanic survival prediction. A representative 120-row subset of the public
 * Kaggle Titanic training data is embedded (pclass, sex, age, sibsp, parch, fare,
 * embarked, survived). Missing ages are imputed with the median.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.tabular.TitanicSurvival" "-Dexec.args=90 16"
 */
public class TitanicSurvival {
    // pclass, sex(0=f,1=m), age, sibsp, parch, fare, embarked(0=C,1=Q,2=S), survived
    static final String ROWS = """
        3,1,22,1,0,7.25,2,1;1,0,38,1,0,71.28,0,1;3,0,26,0,0,7.92,2,1;1,0,35,1,0,53.10,2,1;
        3,1,35,0,0,8.05,2,0;1,1,54,0,0,51.86,2,0;3,1,2,3,1,21.07,2,0;2,0,27,0,2,11.13,2,1;
        3,0,14,1,0,30.07,0,1;3,1,4,1,1,16.70,2,0;1,0,58,0,0,26.55,2,1;3,1,20,0,0,7.23,2,1;
        3,0,39,1,5,31.27,2,0;3,1,14,0,0,7.85,2,0;2,0,55,0,0,16.00,2,1;3,1,2,4,1,29.12,2,0;
        2,1,44,1,0,26.00,2,0;3,0,24,0,0,7.75,1,1;2,1,29,0,0,13.00,2,0;1,0,50,1,0,61.98,0,1;
        3,0,24,0,0,7.23,2,1;3,1,41,0,1,15.50,2,0;2,1,45,0,0,13.00,2,1;1,0,38,1,0,153.46,2,1;
        3,0,25,1,0,7.75,1,1;3,1,47,1,0,9.00,2,0;2,1,62,0,0,10.50,2,0;1,0,32,1,1,110.88,0,1;
        1,0,36,1,0,79.20,0,1;3,1,31,1,0,18.00,2,0;1,0,19,1,0,91.08,2,1;3,1,20,0,0,7.23,2,0;
        2,0,35,0,0,26.00,2,1;3,1,34,0,0,8.05,2,0;3,0,17,4,1,7.78,2,0;2,1,21,1,0,24.00,2,1;
        3,0,33,0,0,7.73,2,1;1,1,50,1,0,55.90,2,0;3,1,23,1,0,15.25,1,0;2,0,40,1,0,26.00,2,1;
        3,1,30,1,0,16.10,2,0;3,0,22,0,0,7.75,1,1;1,1,64,0,0,26.00,2,0;3,0,37,1,1,23.25,2,1;
        2,1,47,1,0,37.00,2,0;1,0,56,0,1,83.16,0,1;3,1,32,1,0,15.50,2,0;3,0,25,0,0,7.65,2,1;
        1,1,30,1,0,93.50,2,0;3,0,18,0,0,7.23,2,1;2,1,42,0,0,13.00,2,0;1,0,49,1,0,89.10,0,1;
        3,0,21,0,0,7.80,2,1;2,1,59,1,0,26.55,2,0;3,1,28,0,0,7.25,2,0;3,0,36,0,0,7.23,2,1;
        2,1,24,2,0,27.00,2,0;1,0,23,1,0,113.27,2,1;3,1,36,1,0,17.40,2,0;3,0,27,0,0,7.75,1,1;
        3,0,29,0,2,23.45,2,1;2,1,32,0,0,26.00,2,0;3,0,19,1,0,7.85,2,1;1,1,48,1,1,52.55,2,0;
        2,0,25,1,0,30.50,2,1;3,1,33,0,0,7.80,2,0;3,1,43,0,1,7.88,2,0;2,0,31,1,0,26.25,2,1;
        3,1,26,0,0,7.88,2,0;3,0,29,1,0,15.85,2,1;1,0,35,0,1,512.33,2,1;2,1,27,1,0,37.00,2,0;
        3,0,22,0,0,7.90,2,1;3,1,45,0,0,7.45,2,0;2,0,34,1,0,23.00,2,1;3,1,49,1,0,16.10,2,0;
        1,0,46,1,0,61.17,0,1;3,1,18,1,0,8.68,2,0;2,0,38,0,0,13.00,2,1;3,0,26,0,0,7.90,2,1;
        3,1,60,1,0,26.55,2,0;3,0,17,1,0,7.17,2,1;1,1,44,0,0,57.98,2,0;3,1,20,0,0,7.05,2,0;
        3,0,23,1,0,13.00,2,1;2,1,53,1,0,30.50,2,0;3,0,30,0,0,7.25,2,1;1,1,65,0,1,109.39,2,0;
        3,0,25,0,0,7.65,2,1;2,1,46,0,0,26.00,2,0;3,0,28,1,0,15.50,2,1;1,0,62,0,0,80.00,2,0;
        3,1,21,1,0,8.05,2,0;3,0,42,1,0,15.17,2,1;2,1,40,1,0,33.00,2,0;3,0,35,0,0,7.23,2,1;
        1,1,58,1,0,146.52,2,0;3,1,24,0,0,7.75,1,0;2,0,39,0,0,26.00,2,1;3,1,37,1,1,23.25,2,0
        """;

    record Row(int pclass, int sex, double age, int sibsp, int parch, double fare, int embarked, int survived) {}

    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 90), batch = Proj.batch(args, 16);
        long t0 = System.currentTimeMillis();

        List rows = new java.util.ArrayList<Row>();
        double ageSum = 0; int ageCount = 0;
        for (String line : ROWS.split(";")) {
            String t = line.trim();
            if (t.isEmpty()) continue;
            String[] f = t.split(",");
            double age = Double.parseDouble(f[2]);
            rows.add(new Row(Integer.parseInt(f[0]), Integer.parseInt(f[1]), age,
                Integer.parseInt(f[3]), Integer.parseInt(f[4]), Double.parseDouble(f[5]),
                Integer.parseInt(f[6]), Integer.parseInt(f[7])));
            if (age > 0) { ageSum += age; ageCount++; }
        }
        double medianAge = ageSum / Math.max(1, ageCount);   // mean-impute

        int n = rows.size();
        Tensor x = new Tensor(n, 9);
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            Row r = (Row) rows.get(i);
            double age = r.age() > 0 ? r.age() : medianAge;
            x.set(i, 0, r.pclass()); x.set(i, 1, r.sex()); x.set(i, 2, age / 100);
            x.set(i, 3, r.sibsp()); x.set(i, 4, r.parch());
            x.set(i, 5, Math.log1p(r.fare()));
            x.set(i, 6, r.embarked() == 0 ? 1 : 0);
            x.set(i, 7, r.embarked() == 1 ? 1 : 0);
            x.set(i, 8, 1);
            y[i] = r.survived();
        }
        double[][] titanicStats = standardizeCols(x, new int[]{0, 1, 2, 3, 4, 5});

        int split = (int) (n * 0.75);
        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(9, 24));
        net.add(new ml.activation.ReLU());
        net.add(new Dropout(0.2));
        net.add(new ml.layer.DenseLayer(24, 2));
        net.add(new ml.activation.Softmax());
        System.out.printf("[P16 Titanic] passengers=%d survivors=%d%n", n,
            java.util.Arrays.stream(y).sum());

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.005))
            .fit(x.sliceRows(0, split), java.util.Arrays.copyOf(y, split), epochs, batch);

        System.out.printf("Train acc: %.1f%% | Test acc: %.1f%%%n",
            100 * acc(net, x.sliceRows(0, split), java.util.Arrays.copyOf(y, split)),
            100 * acc(net, x.sliceRows(split, n), java.util.Arrays.copyOfRange(y, split, n)));

        double[][] demoRaw = {{3, 1, 0.24, 0, 0, Math.log1p(7.5), 0, 0, 1},   // 3rd-class man
                              {1, 0, 0.30, 1, 0, Math.log1p(100), 1, 0, 0}};  // 1st-class woman
        double[][] demo = new double[2][9];
        for (int r = 0; r < 2; r++)
            for (int j : new int[]{0, 1, 2, 3, 4, 5})
                demo[r][j] = (demoRaw[r][j] - titanicStats[0][j]) * titanicStats[1][j];
        for (double[] d : demo) {
            Tensor out = net.forward(Tensor.fromFlat(d, 1, d.length), false);
            System.out.printf("  features %s -> survived=%s (%.0f%%)%n",
                java.util.Arrays.toString(d), out.argmaxRow(0) == 1 ? "YES" : "NO",
                100 * out.data()[out.argmaxRow(0)]);
        }
        Proj.footer("P16 Titanic", t0);
    }

    static double acc(NeuralNetwork net, Tensor x, int[] y) {
        Tensor out = net.forward(x, false);
        int c = 0;
        for (int i = 0; i < x.rows(); i++) if (out.argmaxRow(i) == y[i]) c++;
        return (double) c / x.rows();
    }

    static double[][] standardizeCols(Tensor x, int[] cols) {
        double[][] out = new double[2][cols.length];
        int ci = 0;
        for (int j : cols) {
            double m = 0;
            for (int i = 0; i < x.rows(); i++) m += x.get(i, j);
            m /= x.rows();
            double v = 0;
            for (int i = 0; i < x.rows(); i++) { double d = x.get(i, j) - m; v += d * d; }
            double inv = 1.0 / Math.sqrt(v / Math.max(1, x.rows() - 1) + 1e-9);
            for (int i = 0; i < x.rows(); i++) x.set(i, j, (x.get(i, j) - m) * inv);
            out[0][ci] = m; out[1][ci] = inv; ci++;
        }
        return out;
    }
}




