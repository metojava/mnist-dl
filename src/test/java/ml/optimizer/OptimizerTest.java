package ml.optimizer;

import ml.layer.Layer;
import ml.layer.DenseLayer;
import ml.core.Tensor;
import org.junit.jupiter.api.*;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class OptimizerTest {

    private List<Layer> makeLayers() {
        List<Layer> layers = new ArrayList<>();
        layers.add(new DenseLayer(4, 3));
        layers.add(new DenseLayer(3, 2));
        return layers;
    }

    private void simulateGradient(DenseLayer layer) {
        Tensor input = Tensor.filled(2, layer.inputSize(), 0.1);
        layer.forward(input);
        Tensor grad = Tensor.ones(2, layer.outputSize());
        layer.backward(grad);
    }

    @Test
    void testSGDUpdates() {
        List<Layer> layers = makeLayers();
        DenseLayer l0 = (DenseLayer) layers.get(0);
        float[] origW = l0.weights().data().clone();
        simulateGradient(l0);
        SGD sgd = new SGD(0.01);
        sgd.step(layers, 0.01);
        boolean changed = false;
        for (int i = 0; i < origW.length; i++)
            if (origW[i] != l0.weights().data()[i]) { changed = true; break; }
        assertTrue(changed);
    }

    @Test
    void testSGDMomentum() {
        List<Layer> layers = makeLayers();
        DenseLayer l0 = (DenseLayer) layers.get(0);
        simulateGradient(l0);
        SGD sgd = new SGD(0.01, 0.9);
        float[] w1 = l0.weights().data().clone();
        sgd.step(layers, 0.01);
        float[] w2 = l0.weights().data().clone();
        simulateGradient(l0);
        sgd.step(layers, 0.01);
        assertNotEquals(w1[0], l0.weights().data()[0]);
    }

    @Test
    void testAdamUpdates() {
        List<Layer> layers = makeLayers();
        DenseLayer l0 = (DenseLayer) layers.get(0);
        float[] origW = l0.weights().data().clone();
        simulateGradient(l0);
        Adam adam = new Adam(0.001);
        adam.step(layers, 0.001);
        boolean changed = false;
        for (int i = 0; i < origW.length; i++)
            if (Math.abs(origW[i] - l0.weights().data()[i]) > 1e-10) { changed = true; break; }
        assertTrue(changed);
    }

    @Test
    void testAdamWUpdates() {
        List<Layer> layers = makeLayers();
        DenseLayer l0 = (DenseLayer) layers.get(0);
        simulateGradient(l0);
        AdamW adamW = new AdamW(0.001, 0.01);
        float[] w1 = l0.weights().data().clone();
        adamW.step(layers, 0.001);
        boolean changed = false;
        for (int i = 0; i < w1.length; i++)
            if (Math.abs(w1[i] - l0.weights().data()[i]) > 1e-10) { changed = true; break; }
        assertTrue(changed);
    }

    @Test
    void testRMSPropUpdates() {
        List<Layer> layers = makeLayers();
        DenseLayer l0 = (DenseLayer) layers.get(0);
        simulateGradient(l0);
        RMSProp rms = new RMSProp(0.001);
        float[] w1 = l0.weights().data().clone();
        rms.step(layers, 0.001);
        boolean changed = false;
        for (int i = 0; i < w1.length; i++)
            if (Math.abs(w1[i] - l0.weights().data()[i]) > 1e-10) { changed = true; break; }
        assertTrue(changed);
    }

    @Test
    void testAdaGradUpdates() {
        List<Layer> layers = makeLayers();
        DenseLayer l0 = (DenseLayer) layers.get(0);
        simulateGradient(l0);
        AdaGrad adagrad = new AdaGrad(0.01);
        float[] w1 = l0.weights().data().clone();
        adagrad.step(layers, 0.01);
        boolean changed = false;
        for (int i = 0; i < w1.length; i++)
            if (Math.abs(w1[i] - l0.weights().data()[i]) > 1e-10) { changed = true; break; }
        assertTrue(changed);
    }
}
