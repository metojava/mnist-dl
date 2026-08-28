package projects.common;

import ml.core.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Shared helpers for all demo projects. */
public final class Proj {
    private Proj() {}

    public static int epochs(String[] args, int def) { return args.length > 0 ? Integer.parseInt(args[0]) : def; }
    public static int batch(String[] args, int def) { return args.length > 1 ? Integer.parseInt(args[1]) : def; }
    public static int limit(String[] args, int def) { return args.length > 2 ? Integer.parseInt(args[2]) : def; }

    public static Path out(String project) {
        try { Files.createDirectories(Path.of("outputs", project)); } catch (Exception ignored) {}
        return Path.of("outputs", project);
    }

    public static void header(String title) {
        System.out.println("\n=== " + title + " ===\n");
    }

    public static void footer(String name, long startMs) {
        System.out.printf("%n[%s] finished in %.1fs%n", name, (System.currentTimeMillis() - startMs) / 1000.0);
    }

    /** Fisher-Yates shuffle of paired samples (builds a shuffled copy). */
    public static void shuffle(Tensor x, int[] y, Random rng) {
        int n = x.rows();
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        List<Integer> list = new ArrayList<>(List.of(order));
        java.util.Collections.shuffle(list, rng);
        double[] xd = x.data();
        int cols = x.cols();
        double[] nx = new double[xd.length];
        int[] ny = new int[n];
        for (int i = 0; i < n; i++) {
            int src = list.get(i);
            System.arraycopy(xd, src * cols, nx, i * cols, cols);
            ny[i] = y[src];
        }
        System.arraycopy(nx, 0, xd, 0, xd.length);
        System.arraycopy(ny, 0, y, 0, n);
    }

    /** Standardizes columns to zero mean / unit variance; returns [mean, invStd] params. */
    public static double[][] standardize(Tensor x) {
        int cols = x.cols();
        double[] mean = new double[cols], inv = new double[cols];
        for (int j = 0; j < cols; j++) {
            double s = 0;
            for (int i = 0; i < x.rows(); i++) s += x.get(i, j);
            mean[j] = s / x.rows();
            double v = 0;
            for (int i = 0; i < x.rows(); i++) { double d = x.get(i, j) - mean[j]; v += d * d; }
            inv[j] = 1.0 / Math.sqrt(v / Math.max(1, x.rows() - 1) + 1e-9);
        }
        for (int i = 0; i < x.rows(); i++)
            for (int j = 0; j < cols; j++)
                x.set(i, j, (x.get(i, j) - mean[j]) * inv[j]);
        return new double[][]{mean, inv};
    }

    public interface WindowTarget { double target(double[] window, int wLen); }

    /** Converts a univariate series into flattened sliding windows: rows=(N,T), target per window. */
    public static Tensor windows(double[] series, int T, WindowTarget fn, double[] targetsOut) {
        int n = series.length - T;
        Tensor x = new Tensor(n, T);
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < T; t++) x.set(i, t, series[i + t]);
            targetsOut[i] = fn.target(series, i + T);
        }
        return x;
    }
}
