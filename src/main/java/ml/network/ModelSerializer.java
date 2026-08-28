package ml.network;

import ml.activation.*;
import ml.augmentation.*;
import ml.core.Tensor;
import ml.exception.ModelIOException;
import ml.layer.*;
import ml.optimizer.*;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ModelSerializer {

    // ==================== JSON SAVE ====================

    public static void saveJson(NeuralNetwork net, String filepath) throws ModelIOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n  \"format\": \"mnist-dl/1.0\",\n  \"layers\": [\n");
        List<Layer> layers = net.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            sb.append("    ");
            layerToJson(sb, layers.get(i));
            if (i < layers.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("  ]\n}\n");
        try {
            File file = new File(filepath);
            if (file.getParentFile() != null) file.getParentFile().mkdirs();
            try (FileWriter fw = new FileWriter(filepath)) {
                fw.write(sb.toString());
            }
        } catch (IOException e) {
            throw new ModelIOException("Failed to write JSON: " + e.getMessage(), e);
        }
    }

    private static void layerToJson(StringBuilder sb, Layer layer) {
        sb.append("{ ");
        String type = layer.getClass().getName();
        sb.append("\"type\": \"").append(type).append("\"");

        if (layer instanceof DenseLayer d) {
            sb.append(", \"inputSize\": ").append(d.inputSize());
            sb.append(", \"outputSize\": ").append(d.outputSize());
            sb.append(", \"weights\": ").append(doublesToJson(d.weights().data()));
            sb.append(", \"bias\": ").append(doublesToJson(d.bias().data()));
        } else if (layer instanceof BatchNorm bn) {
            sb.append(", \"featureSize\": ").append(bn.gamma().cols());
            sb.append(", \"gamma\": ").append(doublesToJson(bn.gamma().data()));
            sb.append(", \"beta\": ").append(doublesToJson(bn.beta().data()));
            sb.append(", \"runningMean\": ").append(doublesToJson(bn.runningMean().data()));
            sb.append(", \"runningVar\": ").append(doublesToJson(bn.runningVar().data()));
        } else if (layer instanceof LayerNorm ln) {
            sb.append(", \"featureSize\": ").append(ln.inputSize());
            sb.append(", \"gamma\": ").append(doublesToJson(ln.gamma().data()));
            sb.append(", \"beta\": ").append(doublesToJson(ln.beta().data()));
        } else if (layer instanceof LeakyReLU lr) {
            sb.append(", \"alpha\": ").append(lr.alpha());
        } else if (layer instanceof ELU elu) {
            sb.append(", \"alpha\": ").append(elu.alpha());
        } else if (layer instanceof ml.regularization.Dropout dr) {
            sb.append(", \"rate\": ").append(dr.rate());
        } else if (layer instanceof Conv2D c) {
            sb.append(", \"inChannels\": ").append(c.inputSize());
            sb.append(", \"outChannels\": ").append(c.outputSize());
            sb.append(", \"kernelSize\": ").append(c.kernelSize());
            sb.append(", \"stride\": ").append(c.stride());
            sb.append(", \"padding\": ").append(c.padding());
            for (int i = 0; i < c.kernelCount(); i++) {
                sb.append(", \"kernel").append(i).append("\": ").append(doublesToJson(c.kernel(i).data()));
            }
            sb.append(", \"bias\": ").append(doublesToJson(c.bias().data()));
        } else if (layer instanceof Embedding e) {
            sb.append(", \"vocabSize\": ").append(e.vocabSize());
            sb.append(", \"embeddingDim\": ").append(e.embeddingDim());
            sb.append(", \"weights\": ").append(doublesToJson(e.weights().data()));
        }
        sb.append(" }");
    }

    // ==================== JSON LOAD ====================

    public static NeuralNetwork loadJson(String filepath) throws ModelIOException {
        String json;
        try {
            json = new String(new FileInputStream(filepath).readAllBytes());
        } catch (IOException e) {
            throw new ModelIOException("Failed to read JSON: " + e.getMessage(), e);
        }

        NeuralNetwork net = new NeuralNetwork();
        List<String> layerJsons = extractLayerObjects(json);
        for (String lj : layerJsons) {
            String type = extractString(lj, "type");
            if (type == null) throw new ModelIOException("Missing 'type' in layer JSON");

            Layer layer = switch (type) {
                case "ml.layer.DenseLayer" -> {
                    int in = extractInt(lj, "inputSize");
                    int out = extractInt(lj, "outputSize");
                    DenseLayer d = new DenseLayer(in, out);
                    double[] w = extractDoubles(lj, "weights");
                    double[] b = extractDoubles(lj, "bias");
                    if (w != null) System.arraycopy(w, 0, d.weights().data(), 0, Math.min(w.length, d.weights().size()));
                    if (b != null) System.arraycopy(b, 0, d.bias().data(), 0, Math.min(b.length, d.bias().size()));
                    d.invalidateTranspose();
                    yield d;
                }
                case "ml.layer.BatchNorm" -> {
                    int fs = extractInt(lj, "featureSize");
                    BatchNorm bn = new BatchNorm(fs);
                    double[] g = extractDoubles(lj, "gamma");
                    double[] b2 = extractDoubles(lj, "beta");
                    double[] rm = extractDoubles(lj, "runningMean");
                    double[] rv = extractDoubles(lj, "runningVar");
                    if (g != null) System.arraycopy(g, 0, bn.gamma().data(), 0, fs);
                    if (b2 != null) System.arraycopy(b2, 0, bn.beta().data(), 0, fs);
                    if (rm != null) System.arraycopy(rm, 0, bn.runningMean().data(), 0, fs);
                    if (rv != null) System.arraycopy(rv, 0, bn.runningVar().data(), 0, fs);
                    yield bn;
                }
                case "ml.layer.LayerNorm" -> {
                    int fs = extractInt(lj, "featureSize");
                    LayerNorm ln = new LayerNorm(fs);
                    double[] g = extractDoubles(lj, "gamma");
                    double[] b2 = extractDoubles(lj, "beta");
                    if (g != null) System.arraycopy(g, 0, ln.gamma().data(), 0, fs);
                    if (b2 != null) System.arraycopy(b2, 0, ln.beta().data(), 0, fs);
                    yield ln;
                }
                case "ml.layer.Flatten" -> new Flatten();
                case "ml.activation.ReLU" -> new ReLU();
                case "ml.activation.LeakyReLU" -> {
                    double alpha = extractDouble(lj, "alpha", 0.01);
                    yield new LeakyReLU(alpha);
                }
                case "ml.activation.ELU" -> {
                    double alpha = extractDouble(lj, "alpha", 1.0);
                    yield new ELU(alpha);
                }
                case "ml.activation.Sigmoid" -> new Sigmoid();
                case "ml.activation.Tanh" -> new Tanh();
                case "ml.activation.Softmax" -> new Softmax();
                case "ml.activation.Swish" -> new Swish();
                case "ml.activation.GELU" -> new GELU();
                case "ml.activation.Selu" -> new Selu();
                case "ml.activation.Mish" -> new Mish();
                case "ml.activation.HardSwish" -> new HardSwish();
                case "ml.activation.PReLU" -> {
                    double preluAlpha = extractDouble(lj, "alpha", 0.25);
                    yield new PReLU(preluAlpha);
                }
                case "ml.activation.Softplus" -> new Softplus();
                case "ml.activation.LogSigmoid" -> new LogSigmoid();
                case "ml.regularization.Dropout" -> {
                    double rate = extractDouble(lj, "rate", 0.5);
                    yield new ml.regularization.Dropout(rate);
                }
                case "ml.layer.Conv2D" -> {
                    int inC = extractInt(lj, "inChannels");
                    int outC = extractInt(lj, "outChannels");
                    int ks = extractInt(lj, "kernelSize");
                    int st = extractInt(lj, "stride");
                    int pa = extractInt(lj, "padding");
                    Conv2D c = new Conv2D(inC, outC, ks, st, pa, new java.util.Random());
                    for (int i = 0; i < c.kernelCount(); i++) {
                        double[] k = extractDoubles(lj, "kernel" + i);
                        if (k != null) System.arraycopy(k, 0, c.kernel(i).data(), 0, Math.min(k.length, c.kernel(i).size()));
                    }
                    double[] b2 = extractDoubles(lj, "bias");
                    if (b2 != null) System.arraycopy(b2, 0, c.bias().data(), 0, Math.min(b2.length, c.outputSize()));
                    yield c;
                }
                default -> throw new ModelIOException("Unknown layer type: " + type);
            };
            net.add(layer);
        }
        return net;
    }

    // ==================== BINARY SAVE/LOAD (improved) ====================

    public static void saveBinary(NeuralNetwork net, String filepath) throws ModelIOException {
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(filepath)))) {
            dos.writeUTF("mnist-dl/2.0");
            dos.writeInt(net.getLayers().size());
            for (Layer layer : net.getLayers()) {
                writeLayerBinary(dos, layer);
            }
        } catch (IOException e) {
            throw new ModelIOException("Failed to save model: " + e.getMessage(), e);
        }
    }

    private static void writeLayerBinary(DataOutputStream dos, Layer layer) throws IOException {
        String type = layer.getClass().getName();
        dos.writeUTF(type);

        if (layer instanceof DenseLayer d) {
            dos.writeInt(d.inputSize());
            dos.writeInt(d.outputSize());
            writeDoubles(dos, d.weights().data());
            writeDoubles(dos, d.bias().data());
        } else if (layer instanceof BatchNorm bn) {
            dos.writeInt(bn.gamma().cols());
            writeDoubles(dos, bn.gamma().data());
            writeDoubles(dos, bn.beta().data());
            writeDoubles(dos, bn.runningMean().data());
            writeDoubles(dos, bn.runningVar().data());
        } else if (layer instanceof LayerNorm ln) {
            dos.writeInt(ln.inputSize());
            writeDoubles(dos, ln.gamma().data());
            writeDoubles(dos, ln.beta().data());
        } else if (layer instanceof LeakyReLU lr) {
            dos.writeDouble(lr.alpha());
        } else if (layer instanceof ELU elu) {
            dos.writeDouble(elu.alpha());
        } else if (layer instanceof ml.regularization.Dropout dr) {
            dos.writeDouble(dr.rate());
        } else if (layer instanceof Conv2D c) {
            dos.writeInt(c.inputSize());
            dos.writeInt(c.outputSize());
            dos.writeInt(c.kernelSize());
            dos.writeInt(c.stride());
            dos.writeInt(c.padding());
            for (int i = 0; i < c.kernelCount(); i++) {
                writeDoubles(dos, c.kernel(i).data());
            }
            writeDoubles(dos, c.bias().data());
        } else if (layer instanceof Embedding e) {
            dos.writeInt(e.vocabSize());
            dos.writeInt(e.embeddingDim());
            writeDoubles(dos, e.weights().data());
        } else if (layer instanceof PReLU prelu) {
            dos.writeDouble(prelu.alpha());
        } else if (layer instanceof RNNLayer rnn) {
            dos.writeInt(rnn.getInputSize());
            dos.writeInt(rnn.getOutputSize());
            dos.writeBoolean(rnn.isBidirectional());
            dos.writeBoolean(rnn.isReturnSequences());
        } else if (layer instanceof LSTMLayer lstm) {
            dos.writeInt(lstm.getInputSize());
            dos.writeInt(lstm.getHiddenSize());
            dos.writeBoolean(lstm.isBidirectional());
            dos.writeBoolean(lstm.isReturnSequences());
        } else if (layer instanceof GRULayer gru) {
            dos.writeInt(gru.getInputSize());
            dos.writeInt(gru.getHiddenSize());
            dos.writeBoolean(gru.isBidirectional());
            dos.writeBoolean(gru.isReturnSequences());
        } else if (layer instanceof SelfAttention sa) {
            dos.writeInt(sa.getEmbedDim());
        } else if (layer instanceof MultiHeadAttention mha) {
            dos.writeInt(mha.getEmbedDim());
            dos.writeInt(mha.getNumHeads());
            dos.writeBoolean(mha.isCrossAttention());
        } else if (layer instanceof TransformerBlock tb) {
            dos.writeInt(tb.getEmbedDim());
            dos.writeInt(tb.getNumHeads());
            dos.writeInt(tb.getFeedForwardDim());
            dos.writeBoolean(tb.isCrossAttention());
            dos.writeDouble(0.1);
        } else if (layer instanceof RandomFlip rf) {
            dos.writeUTF(rf.getMode());
            dos.writeDouble(rf.getProbability());
        } else if (layer instanceof RandomRotation rr) {
            dos.writeDouble(rr.getMaxAngleDegrees());
            dos.writeDouble(rr.getProbability());
        } else if (layer instanceof RandomCrop rc) {
            dos.writeInt(rc.getCropSize());
            dos.writeDouble(rc.getProbability());
        } else if (layer instanceof RandomZoom rz) {
            dos.writeDouble(rz.getZoomRange());
            dos.writeDouble(rz.getProbability());
        }
        // Stateless layers (Flatten, ReLU, Sigmoid, Tanh, Softmax, Swish, GELU, Selu) → just type
    }

    public static NeuralNetwork loadBinary(String filepath) throws ModelIOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filepath)))) {
            String magic = dis.readUTF();
            if (!magic.startsWith("mnist-dl/")) {
                throw new ModelIOException("Unknown format: " + magic);
            }
            int version = (int) Double.parseDouble(magic.split("/")[1]);
            NeuralNetwork net = new NeuralNetwork();
            int numLayers = dis.readInt();
            for (int i = 0; i < numLayers; i++) {
                String type = dis.readUTF();
                Layer layer = readLayerBinary(dis, type, version);
                net.add(layer);
            }
            return net;
        } catch (IOException e) {
            throw new ModelIOException("Failed to load model: " + e.getMessage(), e);
        }
    }

    private static Layer readLayerBinary(DataInputStream dis, String type, int version) throws IOException {
        return switch (type) {
            case "ml.layer.DenseLayer" -> {
                int inSize = dis.readInt();
                int outSize = dis.readInt();
                double[] wData = readDoubles(dis);
                double[] bData = readDoubles(dis);
                DenseLayer d = new DenseLayer(inSize, outSize);
                System.arraycopy(wData, 0, d.weights().data(), 0, wData.length);
                System.arraycopy(bData, 0, d.bias().data(), 0, bData.length);
                d.invalidateTranspose();
                yield d;
            }
            case "ml.layer.BatchNorm" -> {
                int fs = dis.readInt();
                BatchNorm bn = new BatchNorm(fs);
                double[] g = readDoubles(dis);
                double[] b = readDoubles(dis);
                double[] rm = readDoubles(dis);
                double[] rv = readDoubles(dis);
                System.arraycopy(g, 0, bn.gamma().data(), 0, fs);
                System.arraycopy(b, 0, bn.beta().data(), 0, fs);
                System.arraycopy(rm, 0, bn.runningMean().data(), 0, fs);
                System.arraycopy(rv, 0, bn.runningVar().data(), 0, fs);
                yield bn;
            }
            case "ml.layer.LayerNorm" -> {
                int fs = dis.readInt();
                LayerNorm ln = new LayerNorm(fs);
                double[] g = readDoubles(dis);
                double[] b = readDoubles(dis);
                System.arraycopy(g, 0, ln.gamma().data(), 0, fs);
                System.arraycopy(b, 0, ln.beta().data(), 0, fs);
                yield ln;
            }
            case "ml.layer.Flatten" -> new Flatten();
            case "ml.activation.ReLU" -> new ReLU();
            case "ml.activation.LeakyReLU" -> new LeakyReLU(dis.readDouble());
            case "ml.activation.ELU" -> new ELU(dis.readDouble());
            case "ml.activation.Sigmoid" -> new Sigmoid();
            case "ml.activation.Tanh" -> new Tanh();
            case "ml.activation.Softmax" -> new Softmax();
            case "ml.activation.Swish" -> new Swish();
            case "ml.activation.GELU" -> new GELU();
            case "ml.activation.Selu" -> new Selu();
            case "ml.activation.Mish" -> new Mish();
            case "ml.activation.HardSwish" -> new HardSwish();
            case "ml.activation.PReLU" -> new PReLU(dis.readDouble());
            case "ml.activation.Softplus" -> new Softplus();
            case "ml.activation.LogSigmoid" -> new LogSigmoid();
            case "ml.regularization.Dropout" -> new ml.regularization.Dropout(dis.readDouble());
            case "ml.layer.RNNLayer" -> {
                int rnnIn = dis.readInt();
                int rnnOut = dis.readInt();
                boolean bidir = dis.readBoolean();
                boolean retSeq = dis.readBoolean();
                yield new RNNLayer(rnnIn, rnnOut, bidir, retSeq, "tanh");
            }
            case "ml.layer.LSTMLayer" -> {
                int lstmIn = dis.readInt();
                int lstmHid = dis.readInt();
                boolean bidir = dis.readBoolean();
                boolean retSeq = dis.readBoolean();
                yield new LSTMLayer(lstmIn, lstmHid, bidir, retSeq);
            }
            case "ml.layer.GRULayer" -> {
                int gruIn = dis.readInt();
                int gruHid = dis.readInt();
                boolean bidir = dis.readBoolean();
                boolean retSeq = dis.readBoolean();
                yield new GRULayer(gruIn, gruHid, bidir, retSeq);
            }
            case "ml.layer.SelfAttention" -> {
                int embedDim = dis.readInt();
                yield new SelfAttention(embedDim);
            }
            case "ml.layer.MultiHeadAttention" -> {
                int mhaEmbed = dis.readInt();
                int heads = dis.readInt();
                boolean cross = dis.readBoolean();
                yield new MultiHeadAttention(mhaEmbed, heads, cross);
            }
            case "ml.layer.TransformerBlock" -> {
                int tbEmbed = dis.readInt();
                int tbHeads = dis.readInt();
                int ffDim = dis.readInt();
                boolean tbCross = dis.readBoolean();
                double tbDrop = dis.readDouble();
                yield new TransformerBlock(tbEmbed, tbHeads, ffDim, tbCross, true, tbDrop);
            }
            case "ml.augmentation.RandomFlip" -> {
                String flipMode = dis.readUTF();
                double flipProb = dis.readDouble();
                yield new RandomFlip(flipMode, flipProb);
            }
            case "ml.augmentation.RandomRotation" -> {
                double maxAngle = dis.readDouble();
                double rotProb = dis.readDouble();
                yield new RandomRotation(maxAngle, rotProb);
            }
            case "ml.augmentation.RandomCrop" -> {
                int cropSize = dis.readInt();
                double cropProb = dis.readDouble();
                yield new RandomCrop(cropSize, cropProb);
            }
            case "ml.augmentation.RandomZoom" -> {
                double zoomRange = dis.readDouble();
                double zoomProb = dis.readDouble();
                yield new RandomZoom(zoomRange, zoomProb);
            }
            case "ml.layer.Conv2D" -> {
                int inC = dis.readInt();
                int outC = dis.readInt();
                int ks = dis.readInt();
                int st = dis.readInt();
                int pa = dis.readInt();
                Conv2D c = new Conv2D(inC, outC, ks, st, pa, new java.util.Random());
                for (int i = 0; i < c.kernelCount(); i++) {
                    double[] k = readDoubles(dis);
                    System.arraycopy(k, 0, c.kernel(i).data(), 0, k.length);
                }
                double[] b2 = readDoubles(dis);
                System.arraycopy(b2, 0, c.bias().data(), 0, b2.length);
                yield c;
            }
            case "ml.layer.Embedding" -> {
                int vocab = dis.readInt();
                int dim = dis.readInt();
                Embedding e = new Embedding(vocab, dim);
                double[] w = readDoubles(dis);
                System.arraycopy(w, 0, e.weights().data(), 0, w.length);
                yield e;
            }
            default -> throw new ModelIOException("Unknown layer type: " + type);
        };
    }

    // ==================== OPTIMIZER STATE SAVE/LOAD ====================

    public static void saveOptimizer(Optimizer opt, String filepath) throws ModelIOException {
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(filepath)))) {
            dos.writeUTF(opt.getClass().getName());
            dos.writeDouble(opt.getLearningRate());

            if (opt instanceof Adam a) {
                a.saveState(dos);
            } else if (opt instanceof AdamW aw) {
                aw.saveState(dos);
            } else if (opt instanceof SGD s) {
                s.saveState(dos);
            } else if (opt instanceof RMSProp r) {
                r.saveState(dos);
            } else if (opt instanceof AdaGrad ag) {
                ag.saveState(dos);
            }
        } catch (IOException e) {
            throw new ModelIOException("Failed to save optimizer: " + e.getMessage(), e);
        }
    }

    public static Optimizer loadOptimizer(String filepath) throws ModelIOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filepath)))) {
            String type = dis.readUTF();
            double lr = dis.readDouble();

            Optimizer opt = switch (type) {
                case "ml.optimizer.Adam" -> {
                    Adam a = new Adam(lr);
                    a.loadState(dis);
                    yield a;
                }
                case "ml.optimizer.AdamW" -> {
                    AdamW aw = new AdamW(lr);
                    aw.loadState(dis);
                    yield aw;
                }
                case "ml.optimizer.SGD" -> {
                    SGD s = new SGD(lr);
                    s.loadState(dis);
                    yield s;
                }
                case "ml.optimizer.RMSProp" -> {
                    RMSProp r = new RMSProp(lr);
                    r.loadState(dis);
                    yield r;
                }
                case "ml.optimizer.AdaGrad" -> {
                    AdaGrad ag = new AdaGrad(lr);
                    ag.loadState(dis);
                    yield ag;
                }
                default -> throw new ModelIOException("Unknown optimizer: " + type);
            };
            return opt;
        } catch (IOException e) {
            throw new ModelIOException("Failed to load optimizer: " + e.getMessage(), e);
        }
    }

    // ==================== HELPERS ====================

    private static StringBuilder doublesToJson(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            if (Double.isNaN(arr[i]) || Double.isInfinite(arr[i])) sb.append("0.0");
            else sb.append(arr[i]);
        }
        sb.append("]");
        return sb;
    }

    public static void writeDoubles(DataOutputStream dos, double[] values) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : values) buf.putDouble(v);
        byte[] bytes = buf.array();
        dos.writeInt(bytes.length);
        dos.write(bytes);
    }

    public static double[] readDoubles(DataInputStream dis) throws IOException {
        int byteLen = dis.readInt();
        byte[] bytes = new byte[byteLen];
        dis.readFully(bytes);
        int count = byteLen / 8;
        double[] result = new double[count];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(result);
        return result;
    }

    // ==================== MINIMAL JSON PARSER ====================

    private static List<String> extractLayerObjects(String json) {
        List<String> result = new ArrayList<>();
        int start = json.indexOf("\"layers\"");
        if (start < 0) return result;
        start = json.indexOf('[', start);
        if (start < 0) return result;
        int depth = 0;
        int objStart = -1;
        for (int i = start + 1; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') { if (depth == 0) objStart = i; depth++; }
            else if (c == '}') { depth--; if (depth == 0 && objStart >= 0) { result.add(json.substring(objStart, i + 1)); objStart = -1; } }
        }
        return result;
    }

    private static String extractString(String json, String key) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
        Matcher m = p.matcher(json);
        return m.find() ? m.group(1) : null;
    }

    private static int extractInt(String json, String key) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?\\d+)");
        Matcher m = p.matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : 0;
    }

    private static double extractDouble(String json, String key, double def) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?[\\d.]+)");
        Matcher m = p.matcher(json);
        return m.find() ? Double.parseDouble(m.group(1)) : def;
    }

    private static double[] extractDoubles(String json, String key) {
        int keyIdx = json.indexOf("\"" + key + "\"");
        if (keyIdx < 0) return null;
        int arrStart = json.indexOf('[', keyIdx);
        if (arrStart < 0) return null;
        int arrEnd = json.indexOf(']', arrStart);
        if (arrEnd < 0) return null;
        String arr = json.substring(arrStart + 1, arrEnd).trim();
        if (arr.isEmpty()) return new double[0];
        String[] parts = arr.split(",");
        double[] result = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Double.parseDouble(parts[i].trim());
        }
        return result;
    }

    // ==================== SAFETENSORS SAVE/LOAD ====================

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    private static String unescapeJson(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char n = s.charAt(i+1);
                if (n == '"') { sb.append('"'); i++; }
                else if (n == '\\') { sb.append('\\'); i++; }
                else if (n == 'n') { sb.append('\n'); i++; }
                else if (n == 'r') { sb.append('\r'); i++; }
                else if (n == 't') { sb.append('\t'); i++; }
                else sb.append(c);
            } else sb.append(c);
        }
        return sb.toString();
    }

    private static String layerConfigJson(Layer layer) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"type\":\"").append(layer.getClass().getName()).append("\"");
        if (layer instanceof DenseLayer d) {
            sb.append(",\"inputSize\":").append(d.inputSize()).append(",\"outputSize\":").append(d.outputSize());
        } else if (layer instanceof BatchNorm bn) {
            sb.append(",\"featureSize\":").append(bn.gamma().cols());
        } else if (layer instanceof LayerNorm ln) {
            sb.append(",\"featureSize\":").append(ln.inputSize());
        } else if (layer instanceof LeakyReLU lr) {
            sb.append(",\"alpha\":").append(lr.alpha());
        } else if (layer instanceof ELU elu) {
            sb.append(",\"alpha\":").append(elu.alpha());
        } else if (layer instanceof ml.regularization.Dropout dr) {
            sb.append(",\"rate\":").append(dr.rate());
        } else if (layer instanceof Conv2D c) {
            sb.append(",\"inChannels\":").append(c.inputSize()).append(",\"outChannels\":").append(c.outputSize());
            sb.append(",\"kernelSize\":").append(c.kernelSize()).append(",\"stride\":").append(c.stride()).append(",\"padding\":").append(c.padding());
        } else if (layer instanceof Embedding e) {
            sb.append(",\"vocabSize\":").append(e.vocabSize()).append(",\"embeddingDim\":").append(e.embeddingDim());
        } else if (layer instanceof PReLU prelu) {
            sb.append(",\"alpha\":").append(prelu.alpha());
        } else if (layer instanceof RNNLayer rnn) {
            sb.append(",\"inputSize\":").append(rnn.getInputSize()).append(",\"outputSize\":").append(rnn.getOutputSize());
            sb.append(",\"bidirectional\":").append(rnn.isBidirectional()).append(",\"returnSequences\":").append(rnn.isReturnSequences());
        } else if (layer instanceof LSTMLayer lstm) {
            sb.append(",\"inputSize\":").append(lstm.getInputSize()).append(",\"hiddenSize\":").append(lstm.getHiddenSize());
            sb.append(",\"bidirectional\":").append(lstm.isBidirectional()).append(",\"returnSequences\":").append(lstm.isReturnSequences());
        } else if (layer instanceof GRULayer gru) {
            sb.append(",\"inputSize\":").append(gru.getInputSize()).append(",\"hiddenSize\":").append(gru.getHiddenSize());
            sb.append(",\"bidirectional\":").append(gru.isBidirectional()).append(",\"returnSequences\":").append(gru.isReturnSequences());
        } else if (layer instanceof SelfAttention sa) {
            sb.append(",\"embedDim\":").append(sa.getEmbedDim());
        } else if (layer instanceof MultiHeadAttention mha) {
            sb.append(",\"embedDim\":").append(mha.getEmbedDim()).append(",\"numHeads\":").append(mha.getNumHeads()).append(",\"isCrossAttention\":").append(mha.isCrossAttention());
        } else if (layer instanceof TransformerBlock tb) {
            sb.append(",\"embedDim\":").append(tb.getEmbedDim()).append(",\"numHeads\":").append(tb.getNumHeads());
            sb.append(",\"feedForwardDim\":").append(tb.getFeedForwardDim()).append(",\"isCrossAttention\":").append(tb.isCrossAttention());
        } else if (layer instanceof RandomFlip rf) {
            sb.append(",\"mode\":\"").append(rf.getMode()).append("\",\"probability\":").append(rf.getProbability());
        } else if (layer instanceof RandomRotation rr) {
            sb.append(",\"maxAngle\":").append(rr.getMaxAngleDegrees()).append(",\"probability\":").append(rr.getProbability());
        } else if (layer instanceof RandomCrop rc) {
            sb.append(",\"cropSize\":").append(rc.getCropSize()).append(",\"probability\":").append(rc.getProbability());
        } else if (layer instanceof RandomZoom rz) {
            sb.append(",\"zoomRange\":").append(rz.getZoomRange()).append(",\"probability\":").append(rz.getProbability());
        }
        sb.append("}");
        return sb.toString();
    }

    private static Layer layerFromConfig(String json) throws ModelIOException {
        String type = extractString(json, "type");
        if (type == null) throw new ModelIOException("Missing type in safetensors config");
        return switch (type) {
            case "ml.layer.DenseLayer" -> new DenseLayer(extractInt(json, "inputSize"), extractInt(json, "outputSize"));
            case "ml.layer.BatchNorm" -> new BatchNorm(extractInt(json, "featureSize"));
            case "ml.layer.LayerNorm" -> new LayerNorm(extractInt(json, "featureSize"));
            case "ml.layer.Flatten" -> new Flatten();
            case "ml.activation.ReLU" -> new ReLU();
            case "ml.activation.LeakyReLU" -> new LeakyReLU(extractDouble(json, "alpha", 0.01));
            case "ml.activation.ELU" -> new ELU(extractDouble(json, "alpha", 1.0));
            case "ml.activation.Sigmoid" -> new Sigmoid();
            case "ml.activation.Tanh" -> new Tanh();
            case "ml.activation.Softmax" -> new Softmax();
            case "ml.activation.Swish" -> new Swish();
            case "ml.activation.GELU" -> new GELU();
            case "ml.activation.Selu" -> new Selu();
            case "ml.activation.Mish" -> new Mish();
            case "ml.activation.HardSwish" -> new HardSwish();
            case "ml.activation.PReLU" -> new PReLU(extractDouble(json, "alpha", 0.25));
            case "ml.activation.Softplus" -> new Softplus();
            case "ml.activation.LogSigmoid" -> new LogSigmoid();
            case "ml.regularization.Dropout" -> new ml.regularization.Dropout(extractDouble(json, "rate", 0.5));
            case "ml.layer.Conv2D" -> new Conv2D(extractInt(json, "inChannels"), extractInt(json, "outChannels"), extractInt(json, "kernelSize"), extractInt(json, "stride"), extractInt(json, "padding"), new java.util.Random());
            case "ml.layer.Embedding" -> new Embedding(extractInt(json, "vocabSize"), extractInt(json, "embeddingDim"));
            case "ml.layer.RNNLayer" -> new RNNLayer(extractInt(json, "inputSize"), extractInt(json, "outputSize"), extractInt(json, "bidirectional") != 0, extractInt(json, "returnSequences") != 0, "tanh");
            case "ml.layer.LSTMLayer" -> new LSTMLayer(extractInt(json, "inputSize"), extractInt(json, "hiddenSize"), extractInt(json, "bidirectional") != 0, extractInt(json, "returnSequences") != 0);
            case "ml.layer.GRULayer" -> new GRULayer(extractInt(json, "inputSize"), extractInt(json, "hiddenSize"), extractInt(json, "bidirectional") != 0, extractInt(json, "returnSequences") != 0);
            case "ml.layer.SelfAttention" -> new SelfAttention(extractInt(json, "embedDim"));
            case "ml.layer.MultiHeadAttention" -> new MultiHeadAttention(extractInt(json, "embedDim"), extractInt(json, "numHeads"), extractInt(json, "isCrossAttention") != 0);
            case "ml.layer.TransformerBlock" -> new TransformerBlock(extractInt(json, "embedDim"), extractInt(json, "numHeads"), extractInt(json, "feedForwardDim"), extractInt(json, "isCrossAttention") != 0, true, 0.1);
            case "ml.augmentation.RandomFlip" -> new RandomFlip(extractString(json, "mode") != null ? extractString(json, "mode") : "horizontal", extractDouble(json, "probability", 0.5));
            case "ml.augmentation.RandomRotation" -> new RandomRotation(extractDouble(json, "maxAngle", 15.0), extractDouble(json, "probability", 0.5));
            case "ml.augmentation.RandomCrop" -> new RandomCrop(extractInt(json, "cropSize"), extractDouble(json, "probability", 0.5));
            case "ml.augmentation.RandomZoom" -> new RandomZoom(extractDouble(json, "zoomRange", 0.2), extractDouble(json, "probability", 0.5));
            default -> throw new ModelIOException("Unknown layer type in safetensors: " + type);
        };
    }

    private static List<String> extractJsonArrayObjects(String jsonArray) {
        List<String> result = new ArrayList<>();
        int depth = 0; int objStart = -1;
        for (int i = 0; i < jsonArray.length(); i++) {
            char c = jsonArray.charAt(i);
            if (c == '{') { if (depth == 0) objStart = i; depth++; }
            else if (c == '}') { depth--; if (depth == 0 && objStart >= 0) { result.add(jsonArray.substring(objStart, i+1)); objStart = -1; } }
        }
        return result;
    }

    public static void saveSafetensors(NeuralNetwork net, String filepath) throws ModelIOException {
        try {
            List<Layer> layers = net.getLayers();
            StringBuilder arch = new StringBuilder("[");
            for (int i = 0; i < layers.size(); i++) {
                arch.append(layerConfigJson(layers.get(i)));
                if (i < layers.size() - 1) arch.append(",");
            }
            arch.append("]");
            String archEsc = escapeJson(arch.toString());

            // Collect tensors
            class Entry { String name; Tensor tensor; int rows; int cols; }
            List<Entry> entries = new ArrayList<>();
            for (int i = 0; i < layers.size(); i++) {
                Layer l = layers.get(i);
                if (l instanceof DenseLayer d) {
                    Entry w = new Entry(); w.name = "layer_" + i + ".DenseLayer.weights"; w.tensor = d.weights(); w.rows = d.weights().rows(); w.cols = d.weights().cols(); entries.add(w);
                    Entry b = new Entry(); b.name = "layer_" + i + ".DenseLayer.bias"; b.tensor = d.bias(); b.rows = 1; b.cols = d.bias().cols(); entries.add(b);
                } else if (l instanceof BatchNorm bn) {
                    Entry g = new Entry(); g.name = "layer_" + i + ".BatchNorm.gamma"; g.tensor = bn.gamma(); g.rows = 1; g.cols = bn.gamma().cols(); entries.add(g);
                    Entry be = new Entry(); be.name = "layer_" + i + ".BatchNorm.beta"; be.tensor = bn.beta(); be.rows = 1; be.cols = bn.beta().cols(); entries.add(be);
                    Entry rm = new Entry(); rm.name = "layer_" + i + ".BatchNorm.runningMean"; rm.tensor = bn.runningMean(); rm.rows = 1; rm.cols = bn.runningMean().cols(); entries.add(rm);
                    Entry rv = new Entry(); rv.name = "layer_" + i + ".BatchNorm.runningVar"; rv.tensor = bn.runningVar(); rv.rows = 1; rv.cols = bn.runningVar().cols(); entries.add(rv);
                } else if (l instanceof LayerNorm ln) {
                    Entry g2 = new Entry(); g2.name = "layer_" + i + ".LayerNorm.gamma"; g2.tensor = ln.gamma(); g2.rows = 1; g2.cols = ln.gamma().cols(); entries.add(g2);
                    Entry be2 = new Entry(); be2.name = "layer_" + i + ".LayerNorm.beta"; be2.tensor = ln.beta(); be2.rows = 1; be2.cols = ln.beta().cols(); entries.add(be2);
                } else if (l instanceof Conv2D c) {
                    for (int k = 0; k < c.kernelCount(); k++) {
                        Entry ke = new Entry(); ke.name = "layer_" + i + ".Conv2D.kernel_" + k; ke.tensor = c.kernel(k); ke.rows = c.kernel(k).rows(); ke.cols = c.kernel(k).cols(); entries.add(ke);
                    }
                    Tensor biasT = c.bias();
                    Entry be3 = new Entry(); be3.name = "layer_" + i + ".Conv2D.bias"; be3.tensor = biasT; be3.rows = 1; be3.cols = biasT.cols(); entries.add(be3);
                } else if (l instanceof Embedding e) {
                    Entry we = new Entry(); we.name = "layer_" + i + ".Embedding.weights"; we.tensor = e.weights(); we.rows = e.weights().rows(); we.cols = e.weights().cols(); entries.add(we);
                }
            }

            // Build header
            StringBuilder header = new StringBuilder("{");
            long offset = 0;
            for (int i = 0; i < entries.size(); i++) {
                Entry e = entries.get(i);
                long byteLen = (long) e.rows * e.cols * 8;
                header.append("\"").append(e.name).append("\":{\"dtype\":\"F64\",\"shape\":[").append(e.rows).append(",").append(e.cols).append("],\"data_offsets\":[").append(offset).append(",").append(offset + byteLen).append("]}");
                header.append(",");
                offset += byteLen;
            }
            header.append("\"__metadata__\":{\"format\":\"mnist-dl/safetensors:1.0\",\"architecture\":\"").append(archEsc).append("\",\"producer\":\"mnist-dl 2.0\"}}");

            byte[] headerBytes = header.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
            long headerLen = headerBytes.length;

            File file = new File(filepath);
            if (file.getParentFile() != null) file.getParentFile().mkdirs();
            try (FileOutputStream fos = new FileOutputStream(filepath);
                 BufferedOutputStream bos = new BufferedOutputStream(fos);
                 DataOutputStream dos = new DataOutputStream(bos)) {
                // 8-byte little-endian header length
                ByteBuffer bb = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(headerLen);
                dos.write(bb.array());
                dos.write(headerBytes);
                // write tensor data little-endian F64
                for (Entry e : entries) {
                    double[] data = e.tensor.data();
                    ByteBuffer tbuf = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN);
                    for (double v : data) tbuf.putDouble(v);
                    dos.write(tbuf.array());
                }
            }
        } catch (IOException e) {
            throw new ModelIOException("Failed to save safetensors: " + e.getMessage(), e);
        }
    }

    public static NeuralNetwork loadSafetensors(String filepath) throws ModelIOException {
        try (FileInputStream fis = new FileInputStream(filepath);
             BufferedInputStream bis = new BufferedInputStream(fis);
             DataInputStream dis = new DataInputStream(bis)) {
            byte[] lenBytes = new byte[8];
            dis.readFully(lenBytes);
            long headerLen = ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
            if (headerLen > 100_000_000) throw new ModelIOException("Safetensors header too large: " + headerLen);
            byte[] headerBytes = new byte[(int) headerLen];
            dis.readFully(headerBytes);
            String header = new String(headerBytes, java.nio.charset.StandardCharsets.UTF_8);

            // Extract architecture
            Pattern metaPat = Pattern.compile("\"__metadata__\"\\s*:\\s*\\{[^}]*\"architecture\"\\s*:\\s*\"((?:\\\\.|[^\"])*)\"");
            Matcher mm = metaPat.matcher(header);
            if (!mm.find()) throw new ModelIOException("Missing architecture in safetensors metadata");
            String archEsc = mm.group(1);
            String archJson = unescapeJson(archEsc);

            List<String> layerJsons = extractJsonArrayObjects(archJson);
            NeuralNetwork net = new NeuralNetwork();
            for (String lj : layerJsons) net.add(layerFromConfig(lj));

            // Parse tensor entries (excluding __metadata__)
            Pattern tPat = Pattern.compile("\"([^\"]+)\"\\s*:\\s*\\{\\s*\"dtype\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\"data_offsets\"\\s*:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]");
            Matcher tm = tPat.matcher(header);
            // read remaining data bytes
            long dataStart = 8 + headerLen;
            File f = new File(filepath);
            long fileLen = f.length();
            long dataLen = fileLen - dataStart;
            byte[] dataBytes = new byte[(int) dataLen];
            dis.readFully(dataBytes);

            while (tm.find()) {
                String name = tm.group(1);
                if (name.equals("__metadata__")) continue;
                String dtype = tm.group(2);
                String shapeStr = tm.group(3);
                long offStart = Long.parseLong(tm.group(4));
                long offEnd = Long.parseLong(tm.group(5));
                int rows = 1, cols = 1;
                if (!shapeStr.trim().isEmpty()) {
                    String[] dims = shapeStr.split(",");
                    if (dims.length >= 1) rows = Integer.parseInt(dims[0].trim());
                    if (dims.length >= 2) cols = Integer.parseInt(dims[1].trim());
                }
                int elem = rows * cols;
                if (!dtype.equals("F64")) throw new ModelIOException("Unsupported dtype " + dtype);
                if (offEnd - offStart != (long) elem * 8) throw new ModelIOException("Offset mismatch for " + name);
                double[] vals = new double[elem];
                ByteBuffer tbuf = ByteBuffer.wrap(dataBytes, (int) offStart, (int) (offEnd - offStart)).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < elem; i++) vals[i] = tbuf.getDouble();

                // Map name to layer
                // name format layer_{idx}.Type.param or layer_{idx}.Type.kernel_{k}
                int dot = name.indexOf('.');
                int layerIdx = Integer.parseInt(name.substring(6, dot));
                String rest = name.substring(dot+1);
                Layer layer = net.getLayers().get(layerIdx);
                if (rest.startsWith("DenseLayer.weights")) {
                    System.arraycopy(vals, 0, ((DenseLayer) layer).weights().data(), 0, vals.length);
                    ((DenseLayer) layer).invalidateTranspose();
                } else if (rest.startsWith("DenseLayer.bias")) {
                    System.arraycopy(vals, 0, ((DenseLayer) layer).bias().data(), 0, vals.length);
                } else if (rest.startsWith("BatchNorm.gamma")) {
                    System.arraycopy(vals, 0, ((BatchNorm) layer).gamma().data(), 0, vals.length);
                } else if (rest.startsWith("BatchNorm.beta")) {
                    System.arraycopy(vals, 0, ((BatchNorm) layer).beta().data(), 0, vals.length);
                } else if (rest.startsWith("BatchNorm.runningMean")) {
                    System.arraycopy(vals, 0, ((BatchNorm) layer).runningMean().data(), 0, vals.length);
                } else if (rest.startsWith("BatchNorm.runningVar")) {
                    System.arraycopy(vals, 0, ((BatchNorm) layer).runningVar().data(), 0, vals.length);
                } else if (rest.startsWith("LayerNorm.gamma")) {
                    System.arraycopy(vals, 0, ((LayerNorm) layer).gamma().data(), 0, vals.length);
                } else if (rest.startsWith("LayerNorm.beta")) {
                    System.arraycopy(vals, 0, ((LayerNorm) layer).beta().data(), 0, vals.length);
                } else if (rest.startsWith("Conv2D.kernel_")) {
                    int k = Integer.parseInt(rest.substring("Conv2D.kernel_".length()));
                    System.arraycopy(vals, 0, ((Conv2D) layer).kernel(k).data(), 0, vals.length);
                } else if (rest.startsWith("Conv2D.bias")) {
                    Tensor b = ((Conv2D) layer).bias();
                    System.arraycopy(vals, 0, b.data(), 0, Math.min(vals.length, b.size()));
                } else if (rest.startsWith("Embedding.weights")) {
                    System.arraycopy(vals, 0, ((Embedding) layer).weights().data(), 0, vals.length);
                }
            }
            return net;
        } catch (IOException e) {
            throw new ModelIOException("Failed to load safetensors: " + e.getMessage(), e);
        }
    }

    // ==================== ONNX SAVE/LOAD (JSON-based ONNX-compatible) ====================

    public static void saveOnnx(NeuralNetwork net, String filepath) throws ModelIOException {
        try {
            List<Layer> layers = net.getLayers();
            StringBuilder sb = new StringBuilder();
            sb.append("{\n  \"ir_version\": 7,\n  \"producer\": \"mnist-dl 2.0\",\n  \"graph\": {\n    \"name\": \"mnist-dl\",\n    \"nodes\": [\n");
            for (int i = 0; i < layers.size(); i++) {
                Layer l = layers.get(i);
                String op, attrs = "";
                if (l instanceof DenseLayer d) { op = "Gemm"; attrs = "\"transB\":1,\"alpha\":1.0,\"beta\":1.0"; }
                else if (l instanceof Conv2D c) { op = "Conv"; attrs = "\"kernel_shape\":[" + c.kernelSize() + "," + c.kernelSize() + "],\"strides\":[" + c.stride() + "," + c.stride() + "],\"pads\":[" + c.padding() + "," + c.padding() + "," + c.padding() + "," + c.padding() + "]"; }
                else if (l instanceof ReLU) op = "Relu";
                else if (l instanceof LeakyReLU) op = "LeakyRelu";
                else if (l instanceof Sigmoid) op = "Sigmoid";
                else if (l instanceof Tanh) op = "Tanh";
                else if (l instanceof Softmax) op = "Softmax";
                else if (l instanceof BatchNorm) op = "BatchNormalization";
                else if (l instanceof LayerNorm) op = "LayerNormalization";
                else if (l instanceof Flatten) op = "Flatten";
                else if (l instanceof MaxPool) op = "MaxPool";
                else if (l instanceof AvgPool) op = "AveragePool";
                else if (l instanceof SelfAttention) op = "Attention";
                else if (l instanceof MultiHeadAttention) op = "MultiHeadAttention";
                else if (l instanceof TransformerBlock) op = "TransformerBlock";
                else if (l instanceof LSTMLayer) op = "LSTM";
                else if (l instanceof GRULayer) op = "GRU";
                else if (l instanceof RNNLayer) op = "RNN";
                else if (l instanceof Embedding) op = "Gather";
                else if (l instanceof ml.regularization.Dropout) op = "Dropout";
                else op = l.getClass().getSimpleName();
                sb.append("      {\"op_type\":\"").append(op).append("\",\"name\":\"layer_").append(i).append("_").append(op).append("\",\"inputs\":[\"x").append(i).append("\"],\"outputs\":[\"x").append(i+1).append("\"]");
                if (!attrs.isEmpty()) sb.append(",\"attrs\":{").append(attrs).append("}");
                sb.append("}");
                if (i < layers.size()-1) sb.append(",");
                sb.append("\n");
            }
            sb.append("    ],\n    \"initializers\": [\n");
            boolean firstInit = true;
            for (int i = 0; i < layers.size(); i++) {
                Layer l = layers.get(i);
                if (l instanceof DenseLayer d) {
                    if (!firstInit) sb.append(",\n");
                    sb.append("      {\"name\":\"layer_").append(i).append("_W\",\"dims\":[").append(d.weights().rows()).append(",").append(d.weights().cols()).append("],\"data\":").append(doublesToJson(d.weights().data())).append("}");
                    sb.append(",\n      {\"name\":\"layer_").append(i).append("_B\",\"dims\":[1,").append(d.bias().cols()).append("],\"data\":").append(doublesToJson(d.bias().data())).append("}");
                    firstInit = false;
                } else if (l instanceof Conv2D c) {
                    for (int k = 0; k < c.kernelCount(); k++) {
                        if (!firstInit) sb.append(",\n");
                        sb.append("      {\"name\":\"layer_").append(i).append("_K").append(k).append("\",\"dims\":[").append(c.kernel(k).rows()).append(",").append(c.kernel(k).cols()).append("],\"data\":").append(doublesToJson(c.kernel(k).data())).append("}");
                        firstInit = false;
                    }
                    if (!firstInit) sb.append(",\n");
                    sb.append("      {\"name\":\"layer_").append(i).append("_B\",\"dims\":[1,").append(c.bias().cols()).append("],\"data\":").append(doublesToJson(c.bias().data())).append("}");
                    firstInit = false;
                } else if (l instanceof BatchNorm bn) {
                    if (!firstInit) sb.append(",\n");
                    sb.append("      {\"name\":\"layer_").append(i).append("_gamma\",\"dims\":[1,").append(bn.gamma().cols()).append("],\"data\":").append(doublesToJson(bn.gamma().data())).append("}");
                    sb.append(",\n      {\"name\":\"layer_").append(i).append("_beta\",\"dims\":[1,").append(bn.beta().cols()).append("],\"data\":").append(doublesToJson(bn.beta().data())).append("}");
                    firstInit = false;
                } else if (l instanceof LayerNorm ln) {
                    if (!firstInit) sb.append(",\n");
                    sb.append("      {\"name\":\"layer_").append(i).append("_gamma\",\"dims\":[1,").append(ln.gamma().cols()).append("],\"data\":").append(doublesToJson(ln.gamma().data())).append("}");
                    sb.append(",\n      {\"name\":\"layer_").append(i).append("_beta\",\"dims\":[1,").append(ln.beta().cols()).append("],\"data\":").append(doublesToJson(ln.beta().data())).append("}");
                    firstInit = false;
                } else if (l instanceof Embedding e) {
                    if (!firstInit) sb.append(",\n");
                    sb.append("      {\"name\":\"layer_").append(i).append("_W\",\"dims\":[").append(e.weights().rows()).append(",").append(e.weights().cols()).append("],\"data\":").append(doublesToJson(e.weights().data())).append("}");
                    firstInit = false;
                }
            }
            sb.append("\n    ],\n    \"layers\": [\n");
            for (int i = 0; i < layers.size(); i++) {
                sb.append("      ").append(layerConfigJson(layers.get(i)));
                if (i < layers.size()-1) sb.append(",");
                sb.append("\n");
            }
            sb.append("    ]\n  }\n}\n");
            File file = new File(filepath);
            if (file.getParentFile() != null) file.getParentFile().mkdirs();
            try (FileWriter fw = new FileWriter(filepath)) { fw.write(sb.toString()); }
        } catch (IOException e) {
            throw new ModelIOException("Failed to save ONNX: " + e.getMessage(), e);
        }
    }

    public static NeuralNetwork loadOnnx(String filepath) throws ModelIOException {
        String json;
        try { json = new String(new FileInputStream(filepath).readAllBytes()); }
        catch (IOException e) { throw new ModelIOException("Failed to read ONNX: " + e.getMessage(), e); }
        int layersIdx = json.indexOf("\"layers\"");
        if (layersIdx < 0) throw new ModelIOException("Missing layers in ONNX");
        int arrStart = json.indexOf('[', layersIdx);
        int depth = 0, end = -1;
        for (int i = arrStart; i < json.length(); i++) {
            if (json.charAt(i) == '[') depth++;
            else if (json.charAt(i) == ']') { depth--; if (depth == 0) { end = i; break; } }
        }
        if (arrStart < 0 || end < 0) throw new ModelIOException("Invalid ONNX layers array");
        String layersArray = json.substring(arrStart, end+1);
        List<String> layerJsons = extractJsonArrayObjects(layersArray);
        NeuralNetwork net = new NeuralNetwork();
        for (String lj : layerJsons) net.add(layerFromConfig(lj));

        // Load initializers by mapping names
        Pattern initPat = Pattern.compile("\"name\"\\s*:\\s*\"layer_(\\d+)_([^\"]+)\"\\s*,\\s*\"dims\"\\s*:\\s*\\[[^\\]]*\\]\\s*,\\s*\"data\"\\s*:\\s*\\[([^\\]]*)\\]");
        Matcher m = initPat.matcher(json);
        while (m.find()) {
            int idx = Integer.parseInt(m.group(1));
            String suffix = m.group(2);
            String dataStr = m.group(3).trim();
            if (idx >= net.getLayers().size()) continue;
            Layer layer = net.getLayers().get(idx);
            double[] vals = new double[0];
            if (!dataStr.isEmpty()) {
                String[] parts = dataStr.split(",");
                vals = new double[parts.length];
                for (int i = 0; i < parts.length; i++) vals[i] = Double.parseDouble(parts[i].trim());
            }
            if (layer instanceof DenseLayer d) {
                if (suffix.equals("W")) { System.arraycopy(vals, 0, d.weights().data(), 0, Math.min(vals.length, d.weights().size())); d.invalidateTranspose(); }
                else if (suffix.equals("B")) System.arraycopy(vals, 0, d.bias().data(), 0, Math.min(vals.length, d.bias().size()));
            } else if (layer instanceof Conv2D c) {
                if (suffix.startsWith("K")) { int k = Integer.parseInt(suffix.substring(1)); System.arraycopy(vals, 0, c.kernel(k).data(), 0, Math.min(vals.length, c.kernel(k).size())); }
                else if (suffix.equals("B")) { Tensor b = c.bias(); System.arraycopy(vals, 0, b.data(), 0, Math.min(vals.length, b.size())); }
            } else if (layer instanceof BatchNorm bn) {
                if (suffix.equals("gamma")) System.arraycopy(vals, 0, bn.gamma().data(), 0, Math.min(vals.length, bn.gamma().size()));
                else if (suffix.equals("beta")) System.arraycopy(vals, 0, bn.beta().data(), 0, Math.min(vals.length, bn.beta().size()));
            } else if (layer instanceof LayerNorm ln) {
                if (suffix.equals("gamma")) System.arraycopy(vals, 0, ln.gamma().data(), 0, Math.min(vals.length, ln.gamma().size()));
                else if (suffix.equals("beta")) System.arraycopy(vals, 0, ln.beta().data(), 0, Math.min(vals.length, ln.beta().size()));
            } else if (layer instanceof Embedding e) {
                if (suffix.equals("W")) System.arraycopy(vals, 0, e.weights().data(), 0, Math.min(vals.length, e.weights().size()));
            }
        }
        return net;
    }
}
