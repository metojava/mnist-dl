package ml.activation;

import ml.core.Tensor;
import ml.layer.DenseLayer;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class ActivationTest {

    @Test
    void testReLU() {
        ReLU relu = new ReLU();
        Tensor input = new Tensor(new double[][]{{-1, 0, 1, 5}});
        Tensor output = relu.forward(input);
        assertEquals(0.0, output.get(0, 0));
        assertEquals(0.0, output.get(0, 1));
        assertEquals(1.0, output.get(0, 2));
        assertEquals(5.0, output.get(0, 3));
    }

    @Test
    void testReLUGradient() {
        ReLU relu = new ReLU();
        Tensor input = new Tensor(new double[][]{{-2, -1, 0, 1, 2}});
        relu.forward(input);
        Tensor grad = Tensor.ones(1, 5);
        Tensor result = relu.backward(grad);
        assertEquals(0.0, result.get(0, 0));
        assertEquals(0.0, result.get(0, 1));
        assertEquals(0.0, result.get(0, 2));
        assertEquals(1.0, result.get(0, 3));
        assertEquals(1.0, result.get(0, 4));
    }

    @Test
    void testLeakyReLU() {
        LeakyReLU lr = new LeakyReLU(0.01);
        Tensor input = new Tensor(new double[][]{{-5, 0, 5}});
        Tensor output = lr.forward(input);
        assertEquals(-0.05, output.get(0, 0), 1e-9);
        assertEquals(0.0, output.get(0, 1), 1e-9);
        assertEquals(5.0, output.get(0, 2), 1e-9);
    }

    @Test
    void testSigmoid() {
        Sigmoid sig = new Sigmoid();
        Tensor input = Tensor.zeros(1, 3);
        Tensor output = sig.forward(input);
        for (int j = 0; j < 3; j++) assertEquals(0.5, output.get(0, j), 1e-9);
    }

    @Test
    void testSigmoidRange() {
        Sigmoid sig = new Sigmoid();
        Tensor input = new Tensor(new double[][]{{-10, 0, 10}});
        Tensor output = sig.forward(input);
        assertTrue(output.get(0, 0) > 0 && output.get(0, 0) < 0.01);
        assertEquals(0.5, output.get(0, 1), 1e-9);
        assertTrue(output.get(0, 2) > 0.99 && output.get(0, 2) <= 1.0);
    }

    @Test
    void testTanh() {
        Tanh tanh = new Tanh();
        Tensor input = Tensor.zeros(1, 2);
        Tensor output = tanh.forward(input);
        assertEquals(0.0, output.get(0, 0), 1e-9);
        assertEquals(0.0, output.get(0, 1), 1e-9);
    }

    @Test
    void testSoftmax() {
        Softmax sm = new Softmax();
        Tensor input = new Tensor(new double[][]{{1, 2, 3}});
        Tensor output = sm.forward(input);
        double sum = 0;
        for (int j = 0; j < 3; j++) {
            assertTrue(output.get(0, j) > 0);
            sum += output.get(0, j);
        }
        assertEquals(1.0, sum, 1e-6);
        assertTrue(output.get(0, 2) > output.get(0, 1));
        assertTrue(output.get(0, 1) > output.get(0, 0));
    }

    @Test
    void testSoftmaxBatch() {
        Softmax sm = new Softmax();
        Tensor input = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor output = sm.forward(input);
        for (int i = 0; i < 2; i++) {
            double sum = 0;
            for (int j = 0; j < 3; j++) sum += output.get(i, j);
            assertEquals(1.0, sum, 1e-6);
        }
    }

    @Test
    void testGELU() {
        GELU gelu = new GELU();
        Tensor input = new Tensor(new double[][]{{0}});
        Tensor output = gelu.forward(input);
        assertEquals(0.0, output.get(0, 0), 1e-6);
    }

    @Test
    void testSwish() {
        Swish swish = new Swish();
        Tensor input = new Tensor(new double[][]{{0}});
        Tensor output = swish.forward(input);
        assertEquals(0.0, output.get(0, 0), 1e-9);
    }

    @Test
    void testELU() {
        ELU elu = new ELU();
        Tensor input = new Tensor(new double[][]{{0, 1}});
        Tensor output = elu.forward(input);
        assertEquals(0.0, output.get(0, 0), 1e-9);
        assertEquals(1.0, output.get(0, 1), 1e-9);
    }

    @Test
    void testSelu() {
        Selu selu = new Selu();
        Tensor input = new Tensor(new double[][]{{0}});
        Tensor output = selu.forward(input);
        assertEquals(0.0, output.get(0, 0), 1e-9);
    }

    @Test
    void testActivationWithDense() {
        DenseLayer dense = new DenseLayer(10, 5);
        ReLU relu = new ReLU();
        Tensor input = Tensor.filled(4, 10, 0.5);
        Tensor out = dense.forward(input);
        Tensor activated = relu.forward(out);
        assertEquals(4, activated.rows());
        assertEquals(5, activated.cols());
    }
}
