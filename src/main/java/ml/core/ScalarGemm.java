package ml.core;

/**
 * Reference scalar GEMM kernels. This is the original hand-blocked code and
 * the fallback used whenever the SIMD kernels are unavailable.
 */
public final class ScalarGemm implements GemmKernels {

    public static final ScalarGemm INSTANCE = new ScalarGemm();

    private ScalarGemm() {
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
                    for (; j + 8 <= jLength; j += 8) {
                        float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        float c4 = 0, c5 = 0, c6 = 0, c7 = 0;
                        for (int k = kk; k < kEnd; k++) {
                            float aValue = ad[aRow + k];
                            int bRow = k * p + jj + j;
                            c0 += aValue * bd[bRow];
                            c1 += aValue * bd[bRow + 1];
                            c2 += aValue * bd[bRow + 2];
                            c3 += aValue * bd[bRow + 3];
                            c4 += aValue * bd[bRow + 4];
                            c5 += aValue * bd[bRow + 5];
                            c6 += aValue * bd[bRow + 6];
                            c7 += aValue * bd[bRow + 7];
                        }
                        rd[resultRow + j] += c0;
                        rd[resultRow + j + 1] += c1;
                        rd[resultRow + j + 2] += c2;
                        rd[resultRow + j + 3] += c3;
                        rd[resultRow + j + 4] += c4;
                        rd[resultRow + j + 5] += c5;
                        rd[resultRow + j + 6] += c6;
                        rd[resultRow + j + 7] += c7;
                    }
                    for (; j + 4 <= jLength; j += 4) {
                        float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        for (int k = kk; k < kEnd; k++) {
                            float aValue = ad[aRow + k];
                            int bRow = k * p + jj + j;
                            c0 += aValue * bd[bRow];
                            c1 += aValue * bd[bRow + 1];
                            c2 += aValue * bd[bRow + 2];
                            c3 += aValue * bd[bRow + 3];
                        }
                        rd[resultRow + j] += c0;
                        rd[resultRow + j + 1] += c1;
                        rd[resultRow + j + 2] += c2;
                        rd[resultRow + j + 3] += c3;
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
                for (; j + 4 <= p; j += 4) {
                    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
                    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
                    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
                    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;
                    for (int k = ib; k < kEnd; k++) {
                        int aCol = k * n;
                        float a0 = ad[aCol + i];
                        float a1 = ad[aCol + i + 1];
                        float a2 = ad[aCol + i + 2];
                        float a3 = ad[aCol + i + 3];
                        int bRow = k * p + j;
                        float b0 = bd[bRow];
                        float b1 = bd[bRow + 1];
                        float b2 = bd[bRow + 2];
                        float b3 = bd[bRow + 3];
                        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
                    }
                    rd[r0p + j] += c00; rd[r0p + j + 1] += c01; rd[r0p + j + 2] += c02; rd[r0p + j + 3] += c03;
                    rd[r1p + j] += c10; rd[r1p + j + 1] += c11; rd[r1p + j + 2] += c12; rd[r1p + j + 3] += c13;
                    rd[r2p + j] += c20; rd[r2p + j + 1] += c21; rd[r2p + j + 2] += c22; rd[r2p + j + 3] += c23;
                    rd[r3p + j] += c30; rd[r3p + j + 1] += c31; rd[r3p + j + 2] += c32; rd[r3p + j + 3] += c33;
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
