package ml.layer;

import ml.core.Tensor;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class DenseLayerTest {

    @Test
    void testForwardShape() {
        DenseLayer layer = new DenseLayer(10, 5);
        Tensor input = Tensor.filled(4, 10, 0.5);
        Tensor output = layer.forward(input);
        assertEquals(4, output.rows());
        assertEquals(5, output.cols());
    }

    @Test
    void testParameterCount() {
        DenseLayer layer = new DenseLayer(10, 5);
        assertEquals(10 * 5 + 5, layer.parameterCount());
    }

    @Test
    void testBackwardGradientShapes() {
        DenseLayer layer = new DenseLayer(8, 4);
        Tensor input = Tensor.filled(2, 8, 1.0);
        layer.forward(input);
        Tensor grad = Tensor.ones(2, 4);
        Tensor inputGrad = layer.backward(grad);
        assertEquals(2, inputGrad.rows());
        assertEquals(8, inputGrad.cols());
        assertNotNull(layer.weightsGradient());
        assertEquals(4, layer.weightsGradient().rows());
        assertEquals(8, layer.weightsGradient().cols());
        assertNotNull(layer.biasGradient());
    }

    @Test
    void testUpdateModifiesWeights() {
        DenseLayer layer = new DenseLayer(4, 3);
        double[] origWeights = layer.weights().data().clone();
        Tensor input = Tensor.filled(1, 4, 1.0);
        layer.forward(input);
        Tensor grad = Tensor.ones(1, 3);
        layer.backward(grad);
        layer.update(0.1);
        boolean changed = false;
        for (int i = 0; i < origWeights.length; i++) {
            if (origWeights[i] != layer.weights().data()[i]) { changed = true; break; }
        }
        assertTrue(changed, "Weights should change after update");
    }

    @Test
    void testLargeLayerParallel() {
        DenseLayer layer = new DenseLayer(256, 128);
        Tensor input = Tensor.filled(64, 256, 0.01);
        Tensor output = layer.forward(input);
        assertEquals(64, output.rows());
        assertEquals(128, output.cols());
        Tensor grad = Tensor.ones(64, 128);
        Tensor inputGrad = layer.backward(grad);
        assertEquals(64, inputGrad.rows());
        assertEquals(256, inputGrad.cols());
        layer.update(0.001);
    }
}
