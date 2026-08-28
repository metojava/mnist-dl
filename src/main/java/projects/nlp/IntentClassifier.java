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
import java.util.Map;

/**
 * P11 - Chatbot intent classification (6 intents, bag-of-words).
 *
 * Run : mvn compile exec:java "-Dexec.mainClass=projects.nlp.IntentClassifier" "-Dexec.args=80 8"
 */
public class IntentClassifier {
    public static void main(String[] args) throws Exception {
        SeedManager.setSeed(42);
        int epochs = Proj.epochs(args, 80), batch = Proj.batch(args, 8);
        long t0 = System.currentTimeMillis();

        List<String> intents = new ArrayList<>(TextCorpus.INTENTS.keySet());
        List<String> texts = new ArrayList<>();
        int[] labels = new int[100];
        int n = 0;
        for (int it = 0; it < intents.size(); it++)
            for (String s : TextCorpus.INTENTS.get(intents.get(it))) { texts.add(s); labels[n++] = it; }

        List<String> aug = new ArrayList<>();
        int[] augLabels = new int[texts.size() * 3];
        Vocab.augment(texts, labels, aug, augLabels, 2, new java.util.Random(9));
        Vocab vocab = new Vocab(aug);
        double[][] feats = Vocab.bowMatrix(aug, vocab);
        Tensor x = Tensor.fromFlat(SentimentAnalyzer.flatten(feats), feats.length, vocab.size);
        Proj.shuffle(x, augLabels, new java.util.Random(9));

        int split = (int) (x.rows() * 0.75);
        NeuralNetwork net = new NeuralNetwork();
        net.add(new ml.layer.DenseLayer(vocab.size, 32));
        net.add(new ml.activation.ReLU());
        net.add(new ml.layer.DenseLayer(32, intents.size()));
        net.add(new ml.activation.Softmax());
        System.out.printf("[P11 Intents] classes=%s samples=%d%n", intents, x.rows());

        new Trainer(net, new CrossEntropyLoss(), new Adam(0.005))
            .fit(x.sliceRows(0, split), java.util.Arrays.copyOf(augLabels, split), epochs, batch);

        System.out.printf("Train acc: %.1f%% | Test acc: %.1f%%%n",
            100 * SentimentAnalyzer.acc(net, x.sliceRows(0, split), java.util.Arrays.copyOf(augLabels, split)),
            100 * SentimentAnalyzer.acc(net, x.sliceRows(split, x.rows()),
                java.util.Arrays.copyOfRange(augLabels, split, x.rows())));

        String[] demo = {"hey how is it going", "will it be cold tomorrow", "add extra cheese to my order",
            "i need help setting this up", "see you tomorrow bye", "thanks that solved it"};
        for (String s : demo) {
            Tensor out = net.forward(Tensor.fromFlat(vocab.bow(s), 1, vocab.size), false);
            System.out.printf("  %-34s -> %s%n", "\"" + s + "\"", intents.get(out.argmaxRow(0)));
        }
        Proj.footer("P11 IntentClassifier", t0);
    }
}


