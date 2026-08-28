package projects.common;

import ml.core.Tensor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Minimal text pipeline: vocabulary, bag-of-words, char n-grams. */
public final class Vocab {
    private final Map<String, Integer> wordToIdx = new HashMap<>();
    private final List<String> idxToWord = new ArrayList<>();
    public final int size;

    public Vocab(List<String> sentences) {
        for (String s : sentences)
            for (String w : tokens(s))
                if (!wordToIdx.containsKey(w)) { wordToIdx.put(w, idxToWord.size()); idxToWord.add(w); }
        size = Math.max(1, idxToWord.size());
    }

    public static String[] tokens(String s) {
        return s.toLowerCase().replaceAll("[^a-z0-9' ]", " ").trim().split("\\s+");
    }

    public int indexOf(String word) { return wordToIdx.getOrDefault(word, -1); }

    public String wordAt(int idx) { return idx >= 0 && idx < idxToWord.size() ? idxToWord.get(idx) : "<unk>"; }

    /** Bag-of-words vector (unigrams + bigrams). */
    public double[] bow(String sentence) {
        double[] v = new double[size];
        String[] t = tokens(sentence);
        for (String w : t) {
            int i = indexOf(w);
            if (i >= 0) v[i] += 1;
        }
        for (int i = 0; i + 1 < t.length; i++) {
            int j = indexOf(t[i] + "_" + t[i + 1]);
            if (j >= 0 && j < size) v[j] += 1;   // bigrams only counted if unigram slot exists
        }
        // L2 normalize
        double norm = 0;
        for (double val : v) norm += val * val;
        norm = Math.sqrt(norm);
        if (norm > 0) for (int i = 0; i < v.length; i++) v[i] /= norm;
        return v;
    }

    /** Hashed character n-gram features over a fixed alphabet. */
    public static double[] charNgrams(String s, int n, int dim) {
        s = s.toLowerCase().replaceAll("[^a-z ]", " ").replaceAll("\\s+", " ");
        double[] v = new double[dim];
        for (int i = 0; i + n <= s.length(); i++) {
            int h = s.substring(i, i + n).hashCode();
            int idx = Math.floorMod(h, dim);
            v[idx] += 1;
        }
        double norm = 0;
        for (double val : v) norm += val;
        if (norm > 0) for (int i = 0; i < dim; i++) v[i] /= norm;
        return v;
    }

    public static double[][] bowMatrix(List<String> sentences, Vocab vocab) {
        double[][] m = new double[sentences.size()][vocab.size];
        for (int i = 0; i < m.length; i++) m[i] = vocab.bow(sentences.get(i));
        return m;
    }

    /** Word-dropout augmentation: original + k corrupted copies per sentence. */
    public static void augment(List<String> textsIn, int[] labelsIn,
                               List<String> textsOut, int[] labelsOut, int copies, java.util.Random rng) {
        for (int i = 0; i < textsIn.size(); i++) {
            String[] t = tokens(textsIn.get(i));
            textsOut.add(textsIn.get(i)); labelsOut[textsOut.size() - 1] = labelsIn[i];
            for (int c = 0; c < copies; c++) {
                StringBuilder sb = new StringBuilder();
                for (String w : t) if (rng.nextDouble() > 0.2) sb.append(w).append(' ');
                textsOut.add(sb.toString().trim()); labelsOut[textsOut.size() - 1] = labelsIn[i];
            }
        }
    }

    /** Builds char-level one-hot sequences: row = flattened T frames of alphabet-size one-hot. */
    public static char[] alphabet() {
        StringBuilder sb = new StringBuilder();
        for (char c = 'a'; c <= 'z'; c++) sb.append(c);
        sb.append(" .,!?'"); 
        return sb.toString().toCharArray();
    }

    public static int charIndex(char c, char[] alphabet) {
        c = Character.toLowerCase(c);
        for (int i = 0; i < alphabet.length; i++) if (alphabet[i] == c) return i;
        return -1;
    }

    public static Tensor charSequences(String text, int T, char[] alphabet, int[] targetsOut, int maxWindows) {
        int V = alphabet.length;
        List<Integer> idxList = new java.util.ArrayList<>();
        for (char c : text.toCharArray()) {
            int i = charIndex(c, alphabet);
            if (i >= 0) idxList.add(i);
        }
        int usable = Math.min(idxList.size() - T, maxWindows);
        Tensor x = new Tensor(usable, T * V);
        for (int n = 0; n < usable; n++)
            for (int t = 0; t < T; t++)
                x.set(n, t * V + idxList.get(n + t), 1.0);
        for (int n = 0; n < usable; n++) targetsOut[n] = idxList.get(n + T);
        return x;
    }
}

