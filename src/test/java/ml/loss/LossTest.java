package ml.loss;

import ml.core.Tensor;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class LossTest {

    @Test
    void testCrossEntropyBasic() {
        CrossEntropyLoss loss = new CrossEntropyLoss();
        Tensor pred = new Tensor(new double[][]{{0.1, 0.7, 0.2}});
        int[] labels = {1};
        double l = loss.forward(pred, labels);
        assertTrue(l > 0, "Cross entropy loss should be positive");
        assertTrue(l < 1.0, "Cross entropy loss should be small for correct prediction");
    }

    @Test
    void testCrossEntropyGradientShape() {
        CrossEntropyLoss loss = new CrossEntropyLoss();
        Tensor pred = new Tensor(new double[][]{{0.2, 0.3, 0.5}, {0.1, 0.6, 0.3}});
        int[] labels = {2, 1};
        Tensor grad = loss.backward(pred, labels);
        assertEquals(2, grad.rows());
        assertEquals(3, grad.cols());
    }

    @Test
    void testMSEBasic() {
        MSELoss loss = new MSELoss();
        Tensor pred = new Tensor(new double[][]{{1.0, 0.0, 0.0}});
        int[] labels = {0};
        double l = loss.forward(pred, labels);
        assertEquals(0.0, l, 1e-9, "Perfect prediction should have 0 loss");
    }

    @Test
    void testMSETensor() {
        MSELoss loss = new MSELoss();
        Tensor pred = new Tensor(new double[][]{{1, 2, 3}});
        Tensor target = new Tensor(new double[][]{{1, 2, 4}});
        double l = loss.forward(pred, target);
        assertEquals(1.0 / 1, l, 1e-9);
    }

    @Test
    void testBinaryCrossEntropy() {
        BinaryCrossEntropyLoss loss = new BinaryCrossEntropyLoss();
        Tensor pred = new Tensor(new double[][]{{0.9}});
        int[] labels = {0};
        double l = loss.forward(pred, labels);
        assertTrue(l > 0);
    }

    @Test
    void testHuberLoss() {
        HuberLoss loss = new HuberLoss(1.0);
        Tensor pred = new Tensor(new double[][]{{2.0, 0.0, 0.0}});
        int[] labels = {0};
        double l = loss.forward(pred, labels);
        assertTrue(l > 0);
    }

    @Test
    void testHingeLoss() {
        HingeLoss loss = new HingeLoss();
        Tensor pred = new Tensor(new double[][]{{2.0, 0.5, 0.1}});
        int[] labels = {0};
        double l = loss.forward(pred, labels);
        assertEquals(0.0, l, 1e-9, "Correct class well ahead should have 0 hinge loss");
    }

    @Test
    void testHingeLossNonZero() {
        HingeLoss loss = new HingeLoss();
        Tensor pred = new Tensor(new double[][]{{0.1, 0.9}});
        int[] labels = {0};
        double l = loss.forward(pred, labels);
        assertTrue(l > 0, "Wrong prediction should have positive hinge loss");
    }
}
