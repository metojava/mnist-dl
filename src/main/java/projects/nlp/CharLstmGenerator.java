package projects.nlp;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import ml.layer.LSTMLayer;
import projects.common.Proj;
import projects.common.TextCorpus;
import projects.common.Vocab;

import java.util.Random;

/**
 * P09 - Character-level language model (Shakespeare excerpt).
 * LSTM over one-hot char windows predicts the next character; then generates text.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.nlp.CharLstmGenerator" "-Dexec.args=12 32"
 */
public class CharLstmGenerator {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 12), batch = Proj.batch(args, 32);
        long t0 = System.currentTimeMillis();

        String text = TextCorpus.SHAKESPEARE_EXCERPT;
        char[] alphabet = Vocab.alphabet();
        int V = alphabet.length, T = 24;

        int[] targets = new int[text.length()];
        Tensor x = Vocab.charSequences(text, T, alphabet, targets, 900);
        System.out.printf("[P09 CharLSTM] corpus=%d chars vocab=%d windows=%d T=%d%n",
            text.length(), V, x.rows(), T);

        NeuralNetwork net = new NeuralNetwork();
        net.add(new LSTMLayer(V, 96));
        net.add(new ml.layer.DenseLayer(96, V));
        net.add(new ml.activation.Softmax());

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.01)).fit(x, targets, epochs, batch);

        // Perplexity-ish check on training windows
        Tensor p = net.forward(x.sliceRows(0, 64), false);
        double ce = 0;
        for (int i = 0; i < 64; i++) ce += -Math.log(Math.max(1e-9, p.get(i, targets[i])));
        System.out.printf("Avg next-char NLL: %.3f (%.1f perplexity)%n", ce / 64, Math.exp(ce / 64));

        // Generate
        Random rng = new Random(7);
        StringBuilder seedText = new StringBuilder("To be, or not");
        StringBuilder gen = new StringBuilder(seedText);
        for (int step = 0; step < 400; step++) {
            String ctx = gen.substring(Math.max(0, gen.length() - T));
            Tensor in = encode(ctx, T, alphabet);
            Tensor probs = net.forward(in, false);
            int next = sample(probs.data(), rng, 0.8);
            gen.append(alphabet[next]);
        }
        System.out.println("\n--- Generated ---\n" + gen + "\n-----------------");
        Proj.footer("P09 CharLSTM", t0);
    }

    private static Tensor encode(String s, int T, char[] alphabet) {
        int V = alphabet.length;
        Tensor row = new Tensor(1, T * V);
        String padded = String.format("%" + T + "s", s).substring(s.length() + T - Math.min(s.length(), T) >= 0 ? 0 : 0);
        // left-pad with spaces if short
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < T) sb.insert(0, ' ');
        if (sb.length() > T) sb = new StringBuilder(sb.substring(sb.length() - T));
        for (int t = 0; t < T; t++) {
            int ci = Vocab.charIndex(sb.charAt(t), alphabet);
            if (ci >= 0) row.set(0, t * V + ci, 1.0);
        }
        return row;
    }

    static int sample(float[] probs, Random rng, double temperature) {
        double[] p = new double[probs.length];
        double max = probs.length > 0 ? probs[0] : 1;
        for (float v : probs) if (v > max) max = v;
        double sum = 0;
        for (int i = 0; i < probs.length; i++) { p[i] = Math.exp((probs[i] - max) / temperature); sum += p[i]; }
        double r = rng.nextDouble() * sum, acc = 0;
        for (int i = 0; i < p.length; i++) { acc += p[i]; if (r <= acc) return i; }
        return p.length - 1;
    }
}
