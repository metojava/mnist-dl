package projects.recommend;

import ml.core.Tensor;
import projects.common.Proj;
import ml.core.SeedManager;
import projects.common.Proj;
import ml.network.NeuralNetwork;
import projects.common.Proj;
import ml.training.*;
import projects.common.Proj;
import ml.optimizer.Adam;
import projects.common.Proj;
import ml.loss.MSELoss;
import projects.common.Proj;

import java.util.ArrayList;
import projects.common.Proj;
import java.util.List;
import projects.common.Proj;
import java.util.Random;
import projects.common.Proj;

/**
 * P19 - Collaborative-filtering movie recommender.
 * Synthetic latent-factor ratings; a two-hot (user+movie) MLP learns the rating
 * function. Demonstrates top-N recommendations for held-out movies of a user.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.recommend.MovieRecommender" "-Dexec.args=8 64"
 */
public class MovieRecommender {
    static final int USERS = 80, MOVIES = 40, K = 8;
    static final String[] GENRES = {"Action", "Comedy", "Drama", "SciFi", "Romance", "Horror",
        "Animation", "Thriller"};

    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 60), batch = Proj.batch(args, 64);
        long t0 = System.currentTimeMillis();

        // Ground-truth preferences
        Random rng = new Random(42);
        double[][] userF = new double[USERS][K];
        double[][] movieF = new double[MOVIES][K];
        for (int u = 0; u < USERS; u++) for (int k = 0; k < K; k++) userF[u][k] = rng.nextGaussian();
        for (int m = 0; m < MOVIES; m++) for (int k = 0; k < K; k++) movieF[m][k] = rng.nextGaussian();

        List<int[]> pairs = new ArrayList<>();
        List<Double> ratings = new ArrayList<>();
        for (int u = 0; u < USERS; u++)
            for (int m = 0; m < MOVIES; m++)
                if (rng.nextDouble() < 0.35) {   // observed ratings (~35% dense)
                    double dot = 0;
                    for (int k = 0; k < K; k++) dot += userF[u][k] * movieF[m][k];
                    double r = Math.max(1, Math.min(5, 3 + 1.1 * dot + 0.4 * rng.nextGaussian()));
                    pairs.add(new int[]{u, m});
                    ratings.add(r);
                }

        int n = pairs.size();
        Tensor x = new Tensor(n, USERS + MOVIES);
        double yMean = ratings.stream().mapToDouble(Double::doubleValue).average().orElse(3);
        double yStd = Math.sqrt(ratings.stream().mapToDouble(r -> (r - yMean) * (r - yMean)).sum() / Math.max(1, n - 1));
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x.set(i, pairs.get(i)[0], 1.0);
            x.set(i, USERS + pairs.get(i)[1], 1.0);
            y[i] = (ratings.get(i) - yMean) / yStd;
        }
        System.out.printf("[P19 Recommender] users=%d movies=%d ratings=%d%n", USERS, MOVIES, n);

        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(USERS + MOVIES, 64));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(64, 32));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(32, 1));
        

        // Train on rating regression (MSE)
        for (int ep = 0; ep < epochs; ep++) {
            double loss = 0; int cnt = 0;
            for (int from = 0; from < n; from += batch) {
                int to = Math.min(from + batch, n);
                Tensor xb = x.sliceRows(from, to);
                Tensor pred = net.forward(xb, true);
                Tensor tgt = yTensor(y, from, to - from);
                loss += mse(pred, y, from, to - from);
                net.backward(pred.subtract(tgt).multiply(2.0 / (to - from)));
                net.update(0.05);
                cnt++;
            }
            if (ep == 0 || (ep + 1) % 5 == 0) System.out.printf("epoch %d: RMSE=%.4f%n", ep + 1, Math.sqrt(loss / cnt));
        }

        // Top-3 recommendations for user 7 among unseen movies
        final int USER = 7;
        record Cand(int movie, double pred) {}
        List<Cand> cands = new ArrayList<>();
        for (int m = 0; m < MOVIES; m++) {
            boolean seen = false;
            for (int[] pr : pairs) if (pr[0] == USER && pr[1] == m) seen = true;
            if (seen) continue;
            Tensor row = new Tensor(1, USERS + MOVIES);
            row.set(0, USER, 1.0); row.set(0, USERS + m, 1.0);
            cands.add(new Cand(m, net.forward(row, false).data()[0] * yStd + yMean));
        }
        cands.sort((a, b2) -> Double.compare(b2.pred(), a.pred()));
        System.out.printf("%nTop-%d recommendations for user %d:%n", Math.min(3, cands.size()), USER);
        for (int i = 0; i < Math.min(3, cands.size()); i++)
            System.out.printf("  movie#%02d [%s] predicted rating %.2f/5%n",
                cands.get(i).movie, GENRES[cands.get(i).movie % GENRES.length], cands.get(i).pred());
        Proj.footer("P19 Recommender", t0);
    }

    static Tensor yTensor(double[] y, int off, int len) {
        return Tensor.fromFlat(java.util.Arrays.copyOfRange(y, off, off + len), len, 1);
    }

    static double mse(Tensor p, double[] y, int off, int len) {
        double s = 0;
        for (int i = 0; i < len; i++) { double d = p.data()[i] - y[off + i]; s += d * d; }
        return s / len;
    }
}




