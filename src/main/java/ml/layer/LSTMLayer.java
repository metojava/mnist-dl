package ml.layer;

import ml.core.Tensor;

import java.util.Random;

/**
 * Long Short-Term Memory layer with true backpropagation-through-time.
 *
 * Input : Tensor(batch, seqLen * inputSize)
 * Output: returnSequences ? Tensor(batch, seqLen * effectiveOut)
 *                         : Tensor(batch, effectiveOut)
 * where effectiveOut = hiddenSize (bidirectional: 2 * hiddenSize).
 */
public class LSTMLayer implements Layer {
    private static final Random INIT_RNG = new Random(1234);

    private final int inputSize;
    private final int hiddenSize;
    private final boolean bidirectional;
    private final boolean returnSequences;

    // ---- Forward-direction parameters ----
    final Tensor wxg, whg, bg;    // cell candidate
    final Tensor wxi, whi, bi;    // input gate
    final Tensor wxf, whf, bf;    // forget gate
    final Tensor wxo, who, bo;    // output gate

    // ---- Backward-direction parameters (bidirectional only) ----
    private Tensor wxgB, whgB, bgB, wxiB, whiB, biB, wxfB, whfB, bfB, wxoB, whoB, boB;

    // ---- Gradients (same layout as parameters) ----
    private double[] gWxg, gWhg, gBg, gWxi, gWhi, gBi, gWxf, gWhf, gBf, gWxo, gWho, gBo;
    private double[] gWxgB, gWhgB, gBgB, gWxiB, gWhiB, gBiB, gWxfB, gWhfB, gBfB, gWxoB, gWhoB, gBoB;

    // ---- Direction-specific trainable bundle ----
    private static final class Dir {
        final Tensor wxg, whg, bg, wxi, whi, bi, wxf, whf, bf, wxo, who, bo;
        final double[] gxg, ghg, gbg, gxi, ghi, gbi, gxf, ghf, gbf, gxo, gho, gbo;
        final boolean forward;

        Dir(Tensor wxg, Tensor whg, Tensor bg, Tensor wxi, Tensor whi, Tensor bi,
            Tensor wxf, Tensor whf, Tensor bf, Tensor wxo, Tensor who, Tensor bo, boolean forward) {
            this.wxg = wxg; this.whg = whg; this.bg = bg;
            this.wxi = wxi; this.whi = whi; this.bi = bi;
            this.wxf = wxf; this.whf = whf; this.bf = bf;
            this.wxo = wxo; this.who = who; this.bo = bo;
            this.gxg = new double[wxg.size()]; this.ghg = new double[whg.size()]; this.gbg = new double[bg.size()];
            this.gxi = new double[wxi.size()]; this.ghi = new double[whi.size()]; this.gbi = new double[bi.size()];
            this.gxf = new double[wxf.size()]; this.ghf = new double[whf.size()]; this.gbf = new double[bf.size()];
            this.gxo = new double[wxo.size()]; this.gho = new double[who.size()]; this.gbo = new double[bo.size()];
            this.forward = forward;
        }
    }

    private final Dir fwd, bwd;

    // ---- Forward cache (per call) ----
    private int seqLen;
    private int batchSize;
    private Cache fc, bc;

    private static final class Cache {
        Tensor[] xt;      // (T) tensors of shape (batch, inputSize)
        Tensor[] i, f, o, g, c, h;   // (T) tensors of shape (batch, hidden)
        Tensor h0, c0;    // initial states (zeros)
        Cache(int T) { xt = new Tensor[T]; i = new Tensor[T]; f = new Tensor[T];
                        o = new Tensor[T]; g = new Tensor[T]; c = new Tensor[T]; h = new Tensor[T]; }
    }

    public LSTMLayer(int inputSize, int hiddenSize, boolean bidirectional, boolean returnSequences) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.bidirectional = bidirectional;
        this.returnSequences = returnSequences;

        this.wxg = init(inputSize, hiddenSize); this.whg = init(hiddenSize, hiddenSize); this.bg = Tensor.zeros(1, hiddenSize);
        this.wxi = init(inputSize, hiddenSize); this.whi = init(hiddenSize, hiddenSize); this.bi = Tensor.zeros(1, hiddenSize);
        this.wxf = init(inputSize, hiddenSize); this.whf = init(hiddenSize, hiddenSize); this.bf = Tensor.filled(1, hiddenSize, 1.0);
        this.wxo = init(inputSize, hiddenSize); this.who = init(hiddenSize, hiddenSize); this.bo = Tensor.zeros(1, hiddenSize);

        this.fwd = new Dir(wxg, whg, bg, wxi, whi, bi, wxf, whf, bf, wxo, who, bo, true);

        if (bidirectional) {
            this.wxgB = init(inputSize, hiddenSize); this.whgB = init(hiddenSize, hiddenSize); this.bgB = Tensor.zeros(1, hiddenSize);
            this.wxiB = init(inputSize, hiddenSize); this.whiB = init(hiddenSize, hiddenSize); this.biB = Tensor.zeros(1, hiddenSize);
            this.wxfB = init(inputSize, hiddenSize); this.whfB = init(hiddenSize, hiddenSize); this.bfB = Tensor.filled(1, hiddenSize, 1.0);
            this.wxoB = init(inputSize, hiddenSize); this.whoB = init(hiddenSize, hiddenSize); this.boB = Tensor.zeros(1, hiddenSize);
            this.bwd = new Dir(wxgB, whgB, bgB, wxiB, whiB, biB, wxfB, whfB, bfB, wxoB, whoB, boB, false);
        } else {
            this.bwd = null;
        }
    }

    private static Tensor init(int rows, int cols) {
        double std = Math.sqrt(2.0 / (rows + cols));
        Tensor t = new Tensor(rows, cols);
        synchronized (INIT_RNG) {
            for (int i = 0; i < t.size(); i++) t.data()[i] = INIT_RNG.nextGaussian() * std;
        }
        return t;
    }

    public LSTMLayer(int inputSize, int hiddenSize) { this(inputSize, hiddenSize, false, false); }
    public LSTMLayer(int inputSize, int hiddenSize, boolean bidirectional) { this(inputSize, hiddenSize, bidirectional, false); }

    // ==================== FORWARD ====================

    @Override
    public Tensor forward(Tensor input, boolean training) {
        batchSize = input.rows();
        seqLen = input.cols() / inputSize;

        fc = runDirection(fwd, input, 0, 1);

        if (!bidirectional) {
            if (returnSequences) {
                Tensor out = Tensor.zeros(batchSize, seqLen * hiddenSize);
                for (int t = 0; t < seqLen; t++)
                    System.arraycopy(fc.h[t].data(), 0, out.data(), t * hiddenSize, batchSize * hiddenSize);
                return out;
            }
            return fc.h[seqLen - 1].copy();
        }

        bc = runDirection(bwd, input, seqLen - 1, -1);

        int eff = hiddenSize * 2;
        Tensor out;
        if (returnSequences) {
            out = Tensor.zeros(batchSize, seqLen * eff);
            for (int t = 0; t < seqLen; t++)
                for (int b = 0; b < batchSize; b++) {
                    int dst = b * seqLen * eff + t * eff;
                    System.arraycopy(fc.h[t].data(), b * hiddenSize, out.data(), dst, hiddenSize);
                    System.arraycopy(bc.h[t].data(), b * hiddenSize, out.data(), dst + hiddenSize, hiddenSize);
                }
        } else {
            out = Tensor.zeros(batchSize, eff);
            for (int b = 0; b < batchSize; b++) {
                System.arraycopy(fc.h[seqLen - 1].data(), b * hiddenSize, out.data(), b * eff, hiddenSize);
                System.arraycopy(bc.h[0].data(), b * hiddenSize, out.data(), b * eff + hiddenSize, hiddenSize);
            }
        }
        return out;
    }

    /** Runs an LSTM over the sequence in the given direction, caching intermediates by timestep. */
    private Cache runDirection(Dir d, Tensor input, int startT, int step) {
        Cache ca = new Cache(seqLen);
        ca.h0 = Tensor.zeros(batchSize, hiddenSize);
        ca.c0 = Tensor.zeros(batchSize, hiddenSize);
        Tensor hPrev = ca.h0, cPrev = ca.c0;

        for (int k = 0; k < seqLen; k++) {
            int t = startT + k * step;
            Tensor xt = sliceCols(input, t * inputSize, (t + 1) * inputSize);
            ca.xt[t] = xt;
            ca.i[t] = sigmoid(Tensor.matmul(xt, d.wxi).add(Tensor.matmul(hPrev, d.whi)), d.bi);
            ca.f[t] = sigmoid(Tensor.matmul(xt, d.wxf).add(Tensor.matmul(hPrev, d.whf)), d.bf);
            ca.o[t] = sigmoid(Tensor.matmul(xt, d.wxo).add(Tensor.matmul(hPrev, d.who)), d.bo);
            ca.g[t] = Tensor.matmul(xt, d.wxg).add(Tensor.matmul(hPrev, d.whg)).apply(Math::tanh);
            ca.c[t] = cPrev.multiply(ca.f[t]).add(ca.i[t].multiply(ca.g[t]));
            ca.h[t] = ca.o[t].multiply(ca.c[t].apply(Math::tanh));
            hPrev = ca.h[t];
            cPrev = ca.c[t];
        }
        return ca;
    }

    private static Tensor sigmoid(Tensor z, Tensor biasRow) {
        double[] d = z.data();
        double[] b = biasRow.data();
        int H = biasRow.cols();
        for (int r = 0; r < z.rows(); r++)
            for (int j = 0; j < H; j++) {
                int i = r * H + j;
                d[i] = 1.0 / (1.0 + Math.exp(-(d[i] + b[j])));
            }
        return z;
    }

    private static Tensor sliceCols(Tensor src, int from, int to) {
        int rows = src.rows();
        int n = to - from;
        double[] data = new double[rows * n];
        for (int r = 0; r < rows; r++)
            System.arraycopy(src.data(), r * src.cols() + from, data, r * n, n);
        return Tensor.fromFlat(data, rows, n);
    }

    // ==================== BACKWARD (BPTT) ====================

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor dInput = Tensor.zeros(batchSize, seqLen * inputSize);
        backwardDirection(fwd, fc, gradient, dInput, 0, 1);
        if (bidirectional) backwardDirection(bwd, bc, gradient, dInput, hiddenSize, -1);
        return dInput;
    }

    /**
     * BPTT over one direction. Caches are indexed by absolute timestep t.
     * @param gradColOffset  column offset of this direction's hidden block within gradient rows
     */
    private void backwardDirection(Dir d, Cache ca, Tensor gradient, Tensor dInput,
                                    int gradColOffset, int step) {
        int H = hiddenSize;
        int T = seqLen;
        int outStride = gradient.cols() / T;

        double[] dhNext = new double[batchSize * H];
        double[] dcNext = new double[batchSize * H];

        for (int k = 0; k < T; k++) {
            int t = step > 0 ? (T - 1 - k) : k;

            // 1. Direct gradient on h_t
            double[] dh = new double[batchSize * H];
            if (returnSequences) {
                for (int b = 0; b < batchSize; b++)
                    for (int j = 0; j < H; j++)
                        dh[b * H + j] = gradient.data()[b * gradient.cols() + t * outStride + gradColOffset + j];
            } else if (bidirectional) {
                boolean endpoint = (step > 0) == (t == T - 1);
                if (endpoint)
                    for (int b = 0; b < batchSize; b++)
                        for (int j = 0; j < H; j++)
                            dh[b * H + j] = gradient.data()[b * gradient.cols() + gradColOffset + j];
            } else if (t == T - 1) {
                for (int idx = 0; idx < dh.length; idx++) dh[idx] = gradient.data()[idx];
            }

            // 2. Previous states
            Tensor hPrev = step > 0 ? (t > 0 ? ca.h[t - 1] : ca.h0) : (t < T - 1 ? ca.h[t + 1] : ca.h0);
            Tensor cPrevT = step > 0 ? (t > 0 ? ca.c[t - 1] : ca.c0) : (t < T - 1 ? ca.c[t + 1] : ca.c0);

            // 3. Gate pre-gradients
            double[] cd = ca.c[t].data();
            double[] od = ca.o[t].data(), fd = ca.f[t].data(), idd = ca.i[t].data(), gd = ca.g[t].data();
            double[] cpd = cPrevT.data();

            double[] di = new double[batchSize * H], df2 = new double[batchSize * H];
            double[] do_ = new double[batchSize * H], dg = new double[batchSize * H];
            double[] dcNew = new double[batchSize * H];

            for (int b = 0; b < batchSize; b++) {
                for (int j = 0; j < H; j++) {
                    int idx = b * H + j;
                    double dhTot = dh[idx] + dhNext[idx];
                    double tc = Math.tanh(cd[idx]);
                    double dc = dhTot * tc + dcNext[idx];

                    do_[idx] = dhTot * tc * od[idx] * (1 - od[idx]);
                    di[idx]  = dc * gd[idx] * idd[idx] * (1 - idd[idx]);
                    df2[idx] = dc * cpd[idx] * fd[idx] * (1 - fd[idx]);
                    dg[idx]  = dc * idd[idx] * (1 - gd[idx] * gd[idx]);
                    dcNew[idx] = dc * fd[idx];

                    dhNext[idx] = 0;
                }
            }

            // 4. Parameter gradients
            Tensor xt = ca.xt[t];
            accGate(d.gxi, xt, di); accGate(d.ghi, hPrev, di); vecAcc(d.gbi, di);
            accGate(d.gxf, xt, df2); accGate(d.ghf, hPrev, df2); vecAcc(d.gbf, df2);
            accGate(d.gxo, xt, do_); accGate(d.gho, hPrev, do_); vecAcc(d.gbo, do_);
            accGate(d.gxg, xt, dg); accGate(d.ghg, hPrev, dg); vecAcc(d.gbg, dg);

            // 5. Recurrent propagation
            propagate(d.whi, di, dhNext);
            propagate(d.whf, df2, dhNext);
            propagate(d.who, do_, dhNext);
            propagate(d.whg, dg, dhNext);
            System.arraycopy(dcNew, 0, dcNext, 0, dcNew.length);

            // 6. Input gradients wrt x_t
            addInputGrad(d.wxi, di, dInput, t);
            addInputGrad(d.wxf, df2, dInput, t);
            addInputGrad(d.wxo, do_, dInput, t);
            addInputGrad(d.wxg, dg, dInput, t);
        }
    }

    private static void accGate(double[] gW, Tensor x, double[] pd) {
        int X = x.cols(), H = pd.length / Math.max(1, x.rows());
        double[] xd = x.data();
        for (int b = 0; b < x.rows(); b++)
            for (int a = 0; a < X; a++) {
                double xv = xd[b * X + a];
                if (xv == 0.0) continue;
                int pBase = b * H, wBase = a * H;
                for (int j = 0; j < H; j++) gW[wBase + j] += xv * pd[pBase + j];
            }
    }

    private static void vecAcc(double[] gB, double[] pd) {
        int H = gB.length;
        for (int b = 0; b < pd.length / H; b++)
            for (int j = 0; j < H; j++) gB[j] += pd[b * H + j];
    }

    private void propagate(Tensor wh, double[] pd, double[] dhNext) {
        int H = hiddenSize;
        double[] wd = wh.data();
        for (int b = 0; b < pd.length / H; b++) {
            int pBase = b * H;
            for (int j = 0; j < H; j++) {
                double sum = 0;
                for (int kk = 0; kk < H; kk++) sum += wd[kk * H + j] * pd[pBase + kk];
                dhNext[pBase + j] += sum;
            }
        }
    }

    private void addInputGrad(Tensor wx, double[] pd, Tensor dInput, int t) {
        int X = inputSize, H = hiddenSize;
        double[] wd = wx.data();
        int colBase = t * X;
        for (int b = 0; b < pd.length / H; b++) {
            int pBase = b * H, dBase = b * dInput.cols() + colBase;
            for (int a = 0; a < X; a++) {
                double sum = 0;
                int wBase = a * H;
                for (int j = 0; j < H; j++) sum += wd[wBase + j] * pd[pBase + j];
                dInput.data()[dBase + a] += sum;
            }
        }
    }

    // ==================== UPDATE ====================

    @Override
    public void update(double learningRate) {
        double scale = learningRate / Math.max(1, batchSize);
        applyGate(fwd, scale);
        if (bwd != null) applyGate(bwd, scale);
    }

    private void applyGate(Dir d, double lrScaled) {
        sgd(d.wxg, d.gxg, lrScaled); sgd(d.whg, d.ghg, lrScaled); vecSgd(d.bg, d.gbg, lrScaled);
        sgd(d.wxi, d.gxi, lrScaled); sgd(d.whi, d.ghi, lrScaled); vecSgd(d.bi, d.gbi, lrScaled);
        sgd(d.wxf, d.gxf, lrScaled); sgd(d.whf, d.ghf, lrScaled); vecSgd(d.bf, d.gbf, lrScaled);
        sgd(d.wxo, d.gxo, lrScaled); sgd(d.who, d.gho, lrScaled); vecSgd(d.bo, d.gbo, lrScaled);
    }

    private static void sgd(Tensor w, double[] g, double lr) {
        double[] wd = w.data();
        for (int i = 0; i < wd.length; i++) { wd[i] -= lr * g[i]; g[i] = 0.0; }
    }

    private static void vecSgd(Tensor biasRow, double[] g, double lr) {
        double[] bd = biasRow.data();
        for (int i = 0; i < bd.length; i++) { bd[i] -= lr * g[i]; g[i] = 0.0; }
    }

    // ==================== META ====================

    @Override
    public int inputSize() { return inputSize; }

    @Override
    public int outputSize() {
        int eff = bidirectional ? hiddenSize * 2 : hiddenSize;
        return returnSequences ? -1 : eff;
    }

    @Override
    public int parameterCount() {
        int per = 4 * (inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize);
        return bidirectional ? per * 2 : per;
    }

    public int getInputSize() { return inputSize; }
    public int getHiddenSize() { return hiddenSize; }
    public boolean isBidirectional() { return bidirectional; }
    public boolean isReturnSequences() { return returnSequences; }
    public Tensor weightsWx() { return wxg; }   public Tensor weightsWh() { return whg; }   public Tensor biasH() { return bg; }
    public Tensor weightsWxi() { return wxi; }  public Tensor weightsWhi() { return whi; }  public Tensor biasHi() { return bi; }
    public Tensor weightsWxf() { return wxf; }  public Tensor weightsWhf() { return whf; }  public Tensor biasHf() { return bf; }
    public Tensor weightsWxo() { return wxo; }  public Tensor weightsWho() { return who; }  public Tensor biasHo() { return bo; }
}
