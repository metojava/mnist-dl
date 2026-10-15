package ml.core;

/**
 * Contract for the two hot GEMM kernels used by {@link Tensor}.
 *
 * Implementations must produce bit-identical results: they may only reorder
 * work in ways that preserve each output element's accumulation order over
 * the reduction dimension. {@link ScalarGemm} is the reference implementation;
 * {@link VectorGemm} is an optional SIMD variant loaded reflectively when the
 * incubator Vector API module is present.
 */
public interface GemmKernels {

    /** C = A*B (A: m x n, B: n x p, C: m x p), rows [r0, r1) of C. */
    void gemmRows(float[] ad, float[] bd, float[] rd, int m, int n, int p, int r0, int r1);

    /** C = A^T*B (A: m x n, B: m x p, C: n x p), rows [r0, r1) of C. */
    void transposedARows(float[] ad, float[] bd, float[] rd, int m, int n, int p, int r0, int r1);
}
