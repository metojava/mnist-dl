package ml.core;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD GEMM kernels built on the incubator Vector API
 * ({@code jdk.incubator.vector}). Loaded reflectively by {@link Tensor} only
 * when that module is resolvable (i.e. the JVM was started with
 * {@code --add-modules jdk.incubator.vector}); otherwise {@link ScalarGemm}
 * is used.
 *
 * The loop blocking (JC/KC for {@code gemmRows}, KB for
 * {@code transposedARows}) and the per-element accumulation order over the
 * reduction dimension are identical to {@link ScalarGemm}, so results are
 * bit-for-bit identical; only the arithmetic is widened to SIMD lanes.
 */
public final class VectorGemm implements GemmKernels {

    public static final VectorGemm INSTANCE = new VectorGemm();

    private static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
    private static final int L = SP.length();

    private VectorGemm() {
    }

    @Override
    public void gemmRows(float[] ad, float[] bd, float[] rd,
                         int m, int n, int p, int r0, int r1) {
        final int JC = 32;
        final int KC = 128;
        for (int jj = 0; jj < p; jj += JC) {
            int jEnd = Math.min(jj + JC, p);
            int jLength = jEnd - jj;
            for (int kk = 0; kk < n; kk += KC) {
                int kEnd = Math.min(kk + KC, n);
                for (int i = r0; i < r1; i++) {
                    int aRow = i * n;
                    int resultRow = i * p + jj;
                    int j = 0;
                    for (; j + L <= jLength; j += L) {
                        FloatVector c = FloatVector.zero(SP);
                        for (int k = kk; k < kEnd; k++) {
                            float aValue = ad[aRow + k];
                            FloatVector bv = FloatVector.fromArray(SP, bd, k * p + jj + j);
                            c = bv.mul(aValue).add(c);
                        }
                        FloatVector.fromArray(SP, rd, resultRow + j).add(c).intoArray(rd, resultRow + j);
                    }
                    for (; j < jLength; j++) {
                        float c = 0;
                        int column = jj + j;
                        for (int k = kk; k < kEnd; k++) c += ad[aRow + k] * bd[k * p + column];
                        rd[resultRow + j] += c;
                    }
                }
            }
        }
    }

    @Override
    public void transposedARows(float[] ad, float[] bd, float[] rd,
                                int m, int n, int p, int r0, int r1) {
        final int KB = 256;
        for (int ib = 0; ib < m; ib += KB) {
            int kEnd = Math.min(ib + KB, m);
            int i = r0;
            for (; i + 4 <= r1; i += 4) {
                int r0p = i * p;
                int r1p = (i + 1) * p;
                int r2p = (i + 2) * p;
                int r3p = (i + 3) * p;
                int j = 0;
                for (; j + L <= p; j += L) {
                    FloatVector c0 = FloatVector.zero(SP);
                    FloatVector c1 = FloatVector.zero(SP);
                    FloatVector c2 = FloatVector.zero(SP);
                    FloatVector c3 = FloatVector.zero(SP);
                    for (int k = ib; k < kEnd; k++) {
                        int aCol = k * n;
                        float a0 = ad[aCol + i];
                        float a1 = ad[aCol + i + 1];
                        float a2 = ad[aCol + i + 2];
                        float a3 = ad[aCol + i + 3];
                        FloatVector bv = FloatVector.fromArray(SP, bd, k * p + j);
                        c0 = bv.mul(a0).add(c0);
                        c1 = bv.mul(a1).add(c1);
                        c2 = bv.mul(a2).add(c2);
                        c3 = bv.mul(a3).add(c3);
                    }
                    FloatVector.fromArray(SP, rd, r0p + j).add(c0).intoArray(rd, r0p + j);
                    FloatVector.fromArray(SP, rd, r1p + j).add(c1).intoArray(rd, r1p + j);
                    FloatVector.fromArray(SP, rd, r2p + j).add(c2).intoArray(rd, r2p + j);
                    FloatVector.fromArray(SP, rd, r3p + j).add(c3).intoArray(rd, r3p + j);
                }
                for (; j < p; j++) {
                    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    for (int k = ib; k < kEnd; k++) {
                        float b = bd[k * p + j];
                        s0 += ad[k * n + i] * b;
                        s1 += ad[k * n + i + 1] * b;
                        s2 += ad[k * n + i + 2] * b;
                        s3 += ad[k * n + i + 3] * b;
                    }
                    rd[r0p + j] += s0;
                    rd[r1p + j] += s1;
                    rd[r2p + j] += s2;
                    rd[r3p + j] += s3;
                }
            }
            for (; i < r1; i++) {
                int resultRow = i * p;
                for (int k = ib; k < kEnd; k++) {
                    float aValue = ad[k * n + i];
                    int bRow = k * p;
                    for (int j = 0; j < p; j++) rd[resultRow + j] += aValue * bd[bRow + j];
                }
            }
        }
    }
}
