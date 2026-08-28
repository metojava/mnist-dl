package ml.core;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class TensorTest {

    @Test
    void testZeros() {
        Tensor t = Tensor.zeros(3, 4);
        assertEquals(3, t.rows());
        assertEquals(4, t.cols());
        assertEquals(12, t.size());
        for (double v : t.data()) assertEquals(0, v);
    }

    @Test
    void testOnes() {
        Tensor t = Tensor.ones(2, 3);
        for (double v : t.data()) assertEquals(1.0, v);
    }

    @Test
    void testFilled() {
        Tensor t = Tensor.filled(5, 5, 3.14);
        for (double v : t.data()) assertEquals(3.14, v, 1e-10);
    }

    @Test
    void testFromDoubleArray() {
        double[][] arr = {{1, 2, 3}, {4, 5, 6}};
        Tensor t = new Tensor(arr);
        assertEquals(2, t.rows());
        assertEquals(3, t.cols());
        assertEquals(1.0, t.get(0, 0));
        assertEquals(6.0, t.get(1, 2));
    }

    @Test
    void testFromFlat() {
        Tensor t = Tensor.fromFlat(new double[]{1, 2, 3, 4}, 2, 2);
        assertEquals(2, t.rows());
        assertEquals(2, t.cols());
        assertEquals(2.0, t.get(0, 1));
        assertEquals(4.0, t.get(1, 1));
    }

    @Test
    void testMatmul() {
        Tensor a = new Tensor(new double[][]{{1, 2}, {3, 4}});
        Tensor b = new Tensor(new double[][]{{5, 6}, {7, 8}});
        Tensor c = Tensor.matmul(a, b);
        assertEquals(2, c.rows());
        assertEquals(2, c.cols());
        assertEquals(19.0, c.get(0, 0), 1e-9); // 1*5+2*7
        assertEquals(22.0, c.get(0, 1), 1e-9); // 1*6+2*8
        assertEquals(43.0, c.get(1, 0), 1e-9); // 3*5+4*7
        assertEquals(50.0, c.get(1, 1), 1e-9); // 3*6+4*8
    }

    @Test
    void testMatmulLarge() {
        int m = 200, k = 300, n = 150;
        Tensor a = Tensor.filled(m, k, 1.0);
        Tensor b = Tensor.filled(k, n, 1.0);
        Tensor c = Tensor.matmul(a, b);
        assertEquals(m, c.rows());
        assertEquals(n, c.cols());
        assertEquals((double) k, c.get(0, 0), 1e-6);
    }

    @Test
    void testTranspose() {
        Tensor t = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor tr = t.transpose();
        assertEquals(3, tr.rows());
        assertEquals(2, tr.cols());
        assertEquals(1.0, tr.get(0, 0));
        assertEquals(4.0, tr.get(0, 1));
        assertEquals(3.0, tr.get(2, 0));
    }

    @Test
    void testAdd() {
        Tensor a = Tensor.ones(2, 2);
        Tensor b = Tensor.filled(2, 2, 2.0);
        Tensor c = a.add(b);
        for (double v : c.data()) assertEquals(3.0, v);
    }

    @Test
    void testSubtract() {
        Tensor a = Tensor.filled(2, 2, 5.0);
        Tensor b = Tensor.ones(2, 2);
        Tensor c = a.subtract(b);
        for (double v : c.data()) assertEquals(4.0, v);
    }

    @Test
    void testMultiplyElementwise() {
        Tensor a = new Tensor(new double[][]{{2, 3}, {4, 5}});
        Tensor b = new Tensor(new double[][]{{10, 20}, {30, 40}});
        Tensor c = a.multiply(b);
        assertEquals(20.0, c.get(0, 0));
        assertEquals(60.0, c.get(0, 1));
        assertEquals(120.0, c.get(1, 0));
        assertEquals(200.0, c.get(1, 1));
    }

    @Test
    void testScalarMultiply() {
        Tensor t = new Tensor(new double[][]{{1, 2}, {3, 4}});
        Tensor result = t.multiply(3.0);
        assertEquals(3.0, result.get(0, 0));
        assertEquals(6.0, result.get(0, 1));
        assertEquals(9.0, result.get(1, 0));
        assertEquals(12.0, result.get(1, 1));
    }

    @Test
    void testSum() {
        Tensor t = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertEquals(21.0, t.sum(), 1e-9);
    }

    @Test
    void testSumLarge() {
        Tensor t = Tensor.filled(1000, 1000, 1.0);
        assertEquals(1_000_000.0, t.sum(), 1e-3);
    }

    @Test
    void testMean() {
        Tensor t = new Tensor(new double[][]{{1, 2}, {3, 4}});
        assertEquals(2.5, t.mean(), 1e-9);
    }

    @Test
    void testMaxMin() {
        Tensor t = new Tensor(new double[][]{{3, 1, 4}, {1, 5, 9}});
        assertEquals(9.0, t.max());
        assertEquals(1.0, t.min());
    }

    @Test
    void testArgmaxRow() {
        Tensor t = new Tensor(new double[][]{{1, 5, 3, 2}});
        assertEquals(1, t.argmaxRow(0));
    }

    @Test
    void testSumRows() {
        Tensor t = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor s = t.sumRows();
        assertEquals(2, s.rows());
        assertEquals(1, s.cols());
        assertEquals(6.0, s.get(0, 0));
        assertEquals(15.0, s.get(1, 0));
    }

    @Test
    void testSumCols() {
        Tensor t = new Tensor(new double[][]{{1, 2}, {3, 4}});
        Tensor s = t.sumCols();
        assertEquals(1, s.rows());
        assertEquals(2, s.cols());
        assertEquals(4.0, s.get(0, 0));
        assertEquals(6.0, s.get(0, 1));
    }

    @Test
    void testApply() {
        Tensor t = new Tensor(new double[][]{{1, 4}, {9, 16}});
        Tensor result = t.apply(Math::sqrt);
        assertEquals(1.0, result.get(0, 0), 1e-9);
        assertEquals(2.0, result.get(0, 1), 1e-9);
        assertEquals(3.0, result.get(1, 0), 1e-9);
        assertEquals(4.0, result.get(1, 1), 1e-9);
    }

    @Test
    void testCopy() {
        Tensor t = new Tensor(new double[][]{{1, 2}, {3, 4}});
        Tensor c = t.copy();
        c.data()[0] = 999;
        assertEquals(1.0, t.get(0, 0), "Copy should not affect original");
    }

    @Test
    void testReadWriteLock() {
        Tensor t = Tensor.zeros(10, 10);
        t.writeLock();
        try { t.data()[0] = 42.0; }
        finally { t.writeUnlock(); }
        t.readLock();
        try { assertEquals(42.0, t.data()[0]); }
        finally { t.readUnlock(); }
    }

    @Test
    void testLargeParallelOps() {
        int n = 5000;
        Tensor a = Tensor.filled(n, 100, 1.0);
        Tensor b = Tensor.filled(n, 100, 2.0);
        Tensor c = a.add(b);
        assertEquals(3.0, c.get(2500, 50), 1e-6);
        Tensor d = a.multiply(b);
        assertEquals(2.0, d.get(2500, 50), 1e-6);
    }

    @Test
    void testLargeParallelMatmul() {
        int size = 500;
        Tensor a = Tensor.filled(size, size, 1.0);
        Tensor b = Tensor.filled(size, size, 2.0);
        Tensor c = Tensor.matmul(a, b);
        assertEquals((double) size * 2, c.get(0, 0), 1e-3);
    }
}
