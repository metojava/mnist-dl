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
 * P08 - Language identification (EN/FR/DE/ES/IT) from character bigrams.
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.nlp.LanguageIdentifier" "-Dexec.args=80 8"
 */
public class LanguageIdentifier {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 80), batch = Proj.batch(args, 8);
        long t0 = System.currentTimeMillis();

        List<String> langs = new ArrayList<>(TextCorpus.LANG_SENTENCES.keySet());
        java.util.Collections.sort(langs);
        List<String> texts = new ArrayList<>();
        int[] labels = new int[100];
        int n = 0;
        for (int L = 0; L < langs.size(); L++)
            for (String s : TextCorpus.LANG_SENTENCES.get(langs.get(L))) { texts.add(s); labels[n++] = L; }

        int dim = 27 * 27;   // char bigram counts over a-z + space
        double[][] feats = bigramFeats(texts, dim);
        Tensor x = Tensor.fromFlat(NewsTopicClassifier.flatten(feats), feats.length, dim);
        Proj.shuffle(x, labels, new java.util.Random(3));

        int split = (int) (x.rows() * 0.75);
        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(dim, 32));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(32, langs.size()));
        net.add(new ml.activation.Softmax());
        System.out.printf("[P08 LangID] sentences=%d languages=%s%n", x.rows(), langs);

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.01))
            .fit(x.sliceRows(0, split), java.util.Arrays.copyOf(labels, split), epochs, batch);

        System.out.printf("Train acc: %.1f%% | Test acc: %.1f%%%n",
            100 * SentimentAnalyzer.acc(net, x.sliceRows(0, split), java.util.Arrays.copyOf(labels, split)),
            100 * SentimentAnalyzer.acc(net, x.sliceRows(split, x.rows()),
                java.util.Arrays.copyOfRange(labels, split, x.rows())));

        String[] demo = {"the weather is nice today", "je voudrais un cafe sil vous plait",
            "ich moechte einen kaffee bitte", "una cerveza fria por favor", "un gelato al limone per favore"};
        for (String s : demo) {
            Tensor out = net.forward(Tensor.fromFlat(bigramFeats(List.of(s), dim)[0], 1, dim), false);
            System.out.printf("  %-38s -> %s%n", "\"" + s + "\"", langs.get(out.argmaxRow(0)));
        }
        Proj.footer("P08 LangID", t0);
    }

    static double[][] bigramFeats(List<String> texts, int dim) {
        double[][] m = new double[texts.size()][dim];
        for (int i = 0; i < texts.size(); i++) {
            String s = texts.get(i).toLowerCase().replaceAll("[^a-z ]", " ").replaceAll("\\s+", " ");
            for (int j = 0; j + 1 < s.length(); j++) {
                int a = s.charAt(j) == ' ' ? 26 : s.charAt(j) - 'a';
                int b2 = s.charAt(j + 1) == ' ' ? 26 : s.charAt(j + 1) - 'a';
                m[i][a * 27 + b2] += 1;
            }
            double norm = 0;
            for (double v : m[i]) norm += v;
            if (norm > 0) for (int j = 0; j < dim; j++) m[i][j] /= norm;
        }
        return m;
    }
}
