package projects.nlp;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import ml.regularization.Dropout;
import projects.common.Proj;
import projects.common.TextCorpus;
import projects.common.Vocab;

import java.util.ArrayList;
import java.util.List;

/**
 * P06 - Sentiment analysis (positive/negative movie reviews).
 * Bag-of-words features over an embedded mini-corpus; MLP classifier.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.nlp.SentimentAnalyzer" "-Dexec.args=40 16"
 */
public class SentimentAnalyzer {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 40), batch = Proj.batch(args, 16);
        long t0 = System.currentTimeMillis();

        List<String> texts = new ArrayList<>();
        int[] labels0 = new int[TextCorpus.POSITIVE.size() + TextCorpus.NEGATIVE.size()];
        int n = 0;
        for (String s : TextCorpus.POSITIVE) { texts.add(s); labels0[n++] = 1; }
        for (String s : TextCorpus.NEGATIVE) { texts.add(s); labels0[n++] = 0; }

        Vocab vocab = new Vocab(texts);
        double[][] feats = Vocab.bowMatrix(texts, vocab);
        Tensor x = Tensor.fromFlat(flatten(feats), feats.length, vocab.size);
        Proj.shuffle(x, labels0, new java.util.Random(42));

        // split 80/20
        int split = (int) (x.rows() * 0.8);
        Tensor trainX = x.sliceRows(0, split), testX = x.sliceRows(split, x.rows());
        int[] trainY = java.util.Arrays.copyOf(labels0, split);
        int[] testY = java.util.Arrays.copyOfRange(labels0, split, x.rows());

        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(vocab.size, 64));
        net.add(new ml.activation.ReLU());
        net.add(new Dropout(0.3));
        net.add(new ml.layer.DenseLayer(64, 2));
        net.add(new ml.activation.Softmax());
        System.out.printf("[P06 Sentiment] vocab=%d train=%d test=%d%n", vocab.size, trainX.rows(), testX.rows());

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.005)).withWeightDecay(new ml.regularization.WeightDecay(0, 1e-3)).fit(trainX, trainY, epochs, batch);

        System.out.printf("Train acc: %.1f%% | Test acc: %.1f%%%n",
            100 * acc(net, trainX, trainY), 100 * acc(net, testX, testY));

        // Demo predictions on unseen sentences
        String[] demo = {
            "a wonderful and moving experience with brilliant acting",
            "boring dull and a complete waste of time",
            "fantastic visuals great story would recommend",
            "terrible script awful pacing i hated it"};
        String[] names = {"NEG", "POS"};
        for (String s : demo) {
            Tensor out = net.forward(Tensor.fromFlat(vocab.bow(s), 1, vocab.size), false);
            System.out.printf("  \"%s\" -> %s (%.0f%%)%n", s, names[out.argmaxRow(0)],
                100 * Math.max(out.get(0, 0), out.get(0, 1)));
        }
        Proj.footer("P06 Sentiment", t0);
    }

    static double acc(NeuralNetwork net, Tensor x, int[] y) {
        Tensor out = net.forward(x, false);
        int c = 0;
        for (int i = 0; i < x.rows(); i++) if (out.argmaxRow(i) == y[i]) c++;
        return (double) c / x.rows();
    }

    static double[] flatten(double[][] m) {
        double[] f = new double[m.length * m[0].length];
        for (int i = 0; i < m.length; i++) System.arraycopy(m[i], 0, f, i * m[0].length, m[0].length);
        return f;
    }
}



