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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * P10 - Next-word prediction (word-level LSTM on the embedded review corpus vocabulary).
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.nlp.NextWordPredictor" "-Dexec.args=15 32"
 */
public class NextWordPredictor {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 15), batch = Proj.batch(args, 32);
        long t0 = System.currentTimeMillis();

        List<String> sentences = new ArrayList<>();
        sentences.addAll(TextCorpus.POSITIVE);
        sentences.addAll(TextCorpus.NEGATIVE);

        Vocab vocab = new Vocab(sentences);
        int V = vocab.size, T = 4;

        // Build (window of T words) -> next word pairs
        List<List<Integer>> windows = new ArrayList<>();
        List<Integer> targets = new ArrayList<>();
        for (String s : sentences) {
            String[] toks = Vocab.tokens(s);
            for (int i = 0; i + T < toks.length; i++) {
                List<Integer> w = new ArrayList<>();
                boolean ok = true;
                for (int j = 0; j <= T; j++) {
                    int ix = vocab.indexOf(toks[i + j]);
                    if (ix < 0) { ok = false; break; }
                    if (j < T) w.add(ix);
                }
                if (ok) { windows.add(w); targets.add(vocab.indexOf(toks[i + T])); }
            }
        }

        Tensor x = new Tensor(targets.size(), T * V);
        for (int n = 0; n < targets.size(); n++)
            for (int t = 0; t < T; t++)
                x.set(n, t * V + windows.get(n).get(t), 1.0);
        int[] y = targets.stream().mapToInt(Integer::intValue).toArray();

        System.out.printf("[P10 NextWord] vocab=%d windows=%d%n", V, x.rows());

        NeuralNetwork net = new NeuralNetwork();
        net.add(new LSTMLayer(V, 128));
        net.add(new ml.layer.DenseLayer(128, V));
        net.add(new ml.activation.Softmax());
        new Trainer(net, new CrossEntropyLoss(), new Adam(0.005)).fit(x, y, epochs, batch);

        Tensor p = net.forward(x.sliceRows(0, Math.min(64, x.rows())), false);
        int top1 = 0;
        for (int i = 0; i < p.rows(); i++) if (p.argmaxRow(i) == y[i]) top1++;
        System.out.printf("Top-1 next-word accuracy (sample): %.1f%%%n", 100.0 * top1 / p.rows());

        // Greedy continuation demo
        String seed = "this movie was";
        StringBuilder sb = new StringBuilder(seed);
        Random rng = new Random(5);
        for (int step = 0; step < 8; step++) {
            String[] toks = Vocab.tokens(sb.toString());
            double[] in = new double[T * V];
            for (int t = 0; t < T; t++) {
                String w = toks[Math.max(0, toks.length - T + t)];
                int ix = vocab.indexOf(w);
                if (ix >= 0) in[t * V + ix] = 1.0;
            }
            Tensor probs = net.forward(Tensor.fromFlat(in, 1, in.length), false);
            int nxt = CharLstmGenerator.sample(probs.data(), rng, 0.7);
            sb.append(' ').append(vocab.wordAt(nxt));
        }
        System.out.println("Continuation: \"" + sb + "\"");
        Proj.footer("P10 NextWord", t0);
    }
}

