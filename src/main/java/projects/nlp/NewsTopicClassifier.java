package projects.nlp;

import ml.core.Tensor;
import ml.core.SeedManager;
import ml.network.NeuralNetwork;
import ml.training.*;
import ml.optimizer.Adam;
import ml.loss.CrossEntropyLoss;
import projects.common.Proj;
import projects.common.TextCorpus;
import projects.common.Vocab;

import java.util.ArrayList;
import java.util.List;

/**
 * P07 - News topic classification (4 topics) with hashed character-trigram features.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.nlp.NewsTopicClassifier" "-Dexec.args=60 8"
 */
public class NewsTopicClassifier {
    static final String[] TOPICS = {"sports", "politics", "technology", "business"};

    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 60), batch = Proj.batch(args, 8);
        long t0 = System.currentTimeMillis();

        List<String> texts = new ArrayList<>();
        int[] labels = new int[100];
        int n = 0;
        for (int t = 0; t < TOPICS.length; t++)
            for (String s : TextCorpus.HEADLINES.get(TOPICS[t])) { texts.add(s); labels[n++] = t; }

        List<String> aug = new ArrayList<>();
        int[] augLabels = new int[texts.size() * 4];
        Vocab.augment(texts, labels, aug, augLabels, 3, new java.util.Random(1));
        Vocab vocab = new Vocab(aug);
        double[][] feats = Vocab.bowMatrix(aug, vocab);
        int dim = vocab.size;
        Tensor x = Tensor.fromFlat(flatten(feats), feats.length, dim);
        Proj.shuffle(x, augLabels, new java.util.Random(1));

        int split = (int) (x.rows() * 0.75);
        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(dim, 48));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(48, TOPICS.length));
        net.add(new ml.activation.Softmax());
        System.out.printf("[P07 NewsTopics] samples=%d trigram-dim=%d%n", x.rows(), dim);

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.005))
            .fit(x.sliceRows(0, split), java.util.Arrays.copyOf(augLabels, split), epochs, batch);

        System.out.printf("Train acc: %.1f%% | Test acc: %.1f%%%n",
            100 * SentimentAnalyzer.acc(net, x.sliceRows(0, split), java.util.Arrays.copyOf(augLabels, split)),
            100 * SentimentAnalyzer.acc(net, x.sliceRows(split, x.rows()),
                java.util.Arrays.copyOfRange(augLabels, split, x.rows())));

        String[] demo = {
            "striker scores hat trick qualifier",
            "senate passes budget reform package",
            "new ai model beats language benchmark",
            "oil prices climb supply concerns persist"};
        for (String s : demo) {
            Tensor out = net.forward(Tensor.fromFlat(vocab.bow(s), 1, dim), false);
            System.out.printf("  %-45s -> %s%n", "\"" + s + "\"", TOPICS[out.argmaxRow(0)]);
        }
        Proj.footer("P07 NewsTopics", t0);
    }

    static double[] flatten(double[][] m) {
        double[] f = new double[m.length * m[0].length];
        for (int i = 0; i < m.length; i++) System.arraycopy(m[i], 0, f, i * m[0].length, m[0].length);
        return f;
    }
}



