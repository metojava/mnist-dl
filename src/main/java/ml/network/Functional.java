package ml.network;

import ml.core.Tensor;
import ml.layer.Layer;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Functional {
    public static class Input {
        private final int size;
        private final String name;
        private final int id;
        private Tensor lastOutput;

        public Input(int size, String name, int id) {
            this.size = size;
            this.name = name;
            this.id = id;
        }

        public Input(int size) {
            this(size, "input_" + System.nanoTime(), (int) (System.nanoTime() % 10000));
        }

        public int getSize() { return size; }
        public String getName() { return name; }
        public int getId() { return id; }

        Tensor forward(Tensor input) {
            this.lastOutput = input;
            return input;
        }
    }

    public static class Node {
        private final Layer layer;
        private final List<Node> inputs;
        private final List<Node> outputs;
        private final String name;
        private final int id;
        private Tensor lastOutput;

        public Node(Layer layer, List<Node> inputs, String name, int id) {
            this.layer = layer;
            this.inputs = inputs != null ? new ArrayList<>(inputs) : new ArrayList<>();
            this.outputs = new ArrayList<>();
            this.name = name;
            this.id = id;

            if (inputs != null) {
                for (Node input : inputs) {
                    input.outputs.add(this);
                }
            }
        }

        public Layer getLayer() { return layer; }
        public List<Node> getInputs() { return inputs; }
        public List<Node> getOutputs() { return outputs; }
        public String getName() { return name; }
        public int getId() { return id; }
    }

    public static class FunctionalModel {
        private final List<Input> inputs;
        private final List<Node> outputs;
        private final List<Node> allNodes;
        private final Map<Integer, Node> nodeMap;
        private final List<Node> sortedNodes;

        public FunctionalModel(List<Input> inputs, List<Node> outputs) {
            this.inputs = new ArrayList<>(inputs);
            this.outputs = new ArrayList<>(outputs);
            this.allNodes = new ArrayList<>();
            this.nodeMap = new HashMap<>();
            this.sortedNodes = topologicalSort();
        }

        public Tensor forward(Map<String, Tensor> inputMap) {
            Map<Integer, Tensor> tensorMap = new HashMap<>();

            for (Input input : inputs) {
                Tensor t = inputMap.get(input.getName());
                if (t == null) throw new IllegalArgumentException("Missing input: " + input.getName());
                tensorMap.put(input.getId(), input.forward(t));
            }

            for (Node node : sortedNodes) {
                List<Tensor> nodeInputs = new ArrayList<>();
                for (Node inputNode : node.inputs) {
                    nodeInputs.add(tensorMap.get(inputNode.getId()));
                }

                Tensor input;
                if (nodeInputs.size() == 1) {
                    input = nodeInputs.get(0);
                } else {
                    input = concatenate(nodeInputs);
                }

                node.lastOutput = node.getLayer().forward(input, true);
                tensorMap.put(node.getId(), node.lastOutput);
            }

            List<Tensor> outputTensors = new ArrayList<>();
            for (Node output : outputs) {
                outputTensors.add(tensorMap.get(output.getId()));
            }

            if (outputTensors.size() == 1) {
                return outputTensors.get(0);
            }
            return concatenate(outputTensors);
        }

        public Tensor forward(Tensor input) {
            if (inputs.size() != 1) {
                throw new IllegalStateException("Model has " + inputs.size() + " inputs, use forward(Map<String, Tensor>)");
            }
            Map<String, Tensor> inputMap = new HashMap<>();
            inputMap.put(inputs.get(0).getName(), input);
            return forward(inputMap);
        }

        public void backward(Tensor gradient) {
            for (int i = sortedNodes.size() - 1; i >= 0; i--) {
                sortedNodes.get(i).getLayer().backward(gradient);
            }
        }

        public void update(double learningRate) {
            for (Node node : sortedNodes) {
                node.getLayer().update(learningRate);
            }
        }

        public int totalParameters() {
            int total = 0;
            for (Node node : allNodes) {
                total += node.getLayer().parameterCount();
            }
            return total;
        }

        public int numberOfLayers() { return allNodes.size(); }

        public List<Node> getTopologicalOrder() { return sortedNodes; }

        public List<Input> getInputs() { return inputs; }
        public List<Node> getOutputs() { return outputs; }

        private List<Node> topologicalSort() {
            Set<Integer> visited = new HashSet<>();
            List<Node> result = new ArrayList<>();

            for (Input input : inputs) {
                visited.add(input.getId());
            }

            for (Node output : outputs) {
                topologicalDFS(output, visited, result);
            }

            return result;
        }

        private void topologicalDFS(Node node, Set<Integer> visited, List<Node> result) {
            if (visited.contains(node.getId())) return;
            visited.add(node.getId());

            for (Node input : node.getInputs()) {
                topologicalDFS(input, visited, result);
            }

            result.add(node);
            nodeMap.put(node.getId(), node);
        }

        private Tensor concatenate(List<Tensor> tensors) {
            if (tensors.size() == 1) return tensors.get(0);

            int totalCols = 0;
            for (Tensor t : tensors) totalCols += t.cols();
            int rows = tensors.get(0).rows();

            Tensor result = Tensor.zeros(rows, totalCols);
            int offset = 0;
            for (Tensor t : tensors) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < t.cols(); c++) {
                        result.data()[r * totalCols + offset + c] = t.get(r, c);
                    }
                }
                offset += t.cols();
            }
            return result;
        }
    }

    private static int nextId = 0;

    public static Input input(int size) {
        return new Input(size);
    }

    public static Input input(int size, String name) {
        return new Input(size, name, nextId++);
    }

    public static Node apply(Layer layer, Node... inputs) {
        List<Node> inputNodes = new ArrayList<>(Arrays.asList(inputs));
        return new Node(layer, inputNodes, layer.getClass().getSimpleName() + "_" + nextId, nextId++);
    }

    public static Node apply(Layer layer, List<Node> inputs) {
        return new Node(layer, inputs, layer.getClass().getSimpleName() + "_" + nextId, nextId++);
    }

    public static FunctionalModel build(List<Input> inputs, List<Node> outputs) {
        return new FunctionalModel(inputs, outputs);
    }

    public static FunctionalModel build(Input input, Node output) {
        return build(List.of(input), List.of(output));
    }

    public static FunctionalModel build(Input input1, Input input2, Node output) {
        return build(List.of(input1, input2), List.of(output));
    }

    public static Node concatenate(Node... nodes) {
        List<Node> inputNodes = Arrays.asList(nodes);
        Layer dummy = new ConcatenateLayer(nodes.length);
        return new Node(dummy, inputNodes, "concatenate_" + nextId, nextId++);
    }

    public static Node add(Node... nodes) {
        List<Node> inputNodes = Arrays.asList(nodes);
        Layer dummy = new AddLayer();
        return new Node(dummy, inputNodes, "add_" + nextId, nextId++);
    }

    private static class ConcatenateLayer implements Layer {
        private final int numInputs;

        public ConcatenateLayer(int numInputs) {
            this.numInputs = numInputs;
        }

        @Override
        public Tensor forward(Tensor input, boolean training) { return input; }

        @Override
        public Tensor backward(Tensor gradient) { return gradient; }

        @Override
        public void update(double learningRate) {}

        @Override
        public int inputSize() { return -1; }

        @Override
        public int outputSize() { return -1; }

        @Override
        public int parameterCount() { return 0; }
    }

    private static class AddLayer implements Layer {
        @Override
        public Tensor forward(Tensor input, boolean training) { return input; }

        @Override
        public Tensor backward(Tensor gradient) { return gradient; }

        @Override
        public void update(double learningRate) {}

        @Override
        public int inputSize() { return -1; }

        @Override
        public int outputSize() { return -1; }

        @Override
        public int parameterCount() { return 0; }
    }
}
