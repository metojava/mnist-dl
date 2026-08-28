package ml.util;

import ml.core.Tensor;
import ml.exception.ModelIOException;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;

public class DataReader {

    public record DataResult(Tensor features, int[] labels, String[] columnNames) {}

    public enum FileFormat {
        CSV, TSV, JSON, TXT, PARQUET, NUMPY, NUMPY_ZIP, SAFETENSORS, ARFF, LIBSVM, UNKNOWN
    }

    // ==================== UNIVERSAL READ ====================

    public static DataResult read(String filepath) throws ModelIOException {
        FileFormat format = detectFormat(filepath);
        return switch (format) {
            case CSV -> readCsv(filepath);
            case TSV -> readTsv(filepath);
            case JSON -> readJson(filepath);
            case TXT -> readTxt(filepath);
            case PARQUET -> readParquet(filepath);
            case NUMPY -> readNumpy(filepath);
            case NUMPY_ZIP -> readNumpyZip(filepath);
            case SAFETENSORS -> readSafetensors(filepath);
            case ARFF -> readArff(filepath);
            case LIBSVM -> readLibSvm(filepath);
            default -> throw new ModelIOException("Unsupported file format: " + filepath);
        };
    }

    public static FileFormat detectFormat(String filepath) {
        String lower = filepath.toLowerCase();
        if (lower.endsWith(".csv")) return FileFormat.CSV;
        if (lower.endsWith(".tsv")) return FileFormat.TSV;
        if (lower.endsWith(".json")) return FileFormat.JSON;
        if (lower.endsWith(".txt") || lower.endsWith(".text")) return FileFormat.TXT;
        if (lower.endsWith(".parquet") || lower.endsWith(".pq")) return FileFormat.PARQUET;
        if (lower.endsWith(".npy")) return FileFormat.NUMPY;
        if (lower.endsWith(".npz")) return FileFormat.NUMPY_ZIP;
        if (lower.endsWith(".safetensors") || lower.endsWith(".st")) return FileFormat.SAFETENSORS;
        if (lower.endsWith(".arff")) return FileFormat.ARFF;
        if (lower.endsWith(".libsvm") || lower.endsWith(".svm")) return FileFormat.LIBSVM;
        return FileFormat.UNKNOWN;
    }

    // ==================== CSV ====================

    public static DataResult readCsv(String filepath) throws ModelIOException {
        return readCsv(filepath, true, ",", -1);
    }

    public static DataResult readCsv(String filepath, boolean hasHeader) throws ModelIOException {
        return readCsv(filepath, hasHeader, ",", -1);
    }

    public static DataResult readCsv(String filepath, boolean hasHeader, String delimiter) throws ModelIOException {
        return readCsv(filepath, hasHeader, delimiter, -1);
    }

    public static DataResult readCsv(String filepath, boolean hasHeader, String delimiter, int labelColumn) throws ModelIOException {
        try {
            List<String[]> rows = new ArrayList<>();
            String[] headers = null;

            try (BufferedReader br = new BufferedReader(new FileReader(filepath, StandardCharsets.UTF_8))) {
                String line;
                boolean firstLine = true;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;

                    if (firstLine && hasHeader) {
                        headers = parseLine(line, delimiter);
                        firstLine = false;
                        continue;
                    }

                    String[] parts = parseLine(line, delimiter);
                    rows.add(parts);
                    firstLine = false;
                }
            }

            if (rows.isEmpty()) throw new ModelIOException("No data found in: " + filepath);
            int numCols = rows.get(0).length;
            if (labelColumn >= numCols) throw new ModelIOException("Label column " + labelColumn + " out of range (" + numCols + " cols)");

            if (headers == null) {
                headers = new String[numCols];
                for (int i = 0; i < numCols; i++) headers[i] = "col_" + i;
            }

            int featureCols = labelColumn >= 0 ? numCols - 1 : numCols;
            double[][] featureData = new double[rows.size()][featureCols];
            int[] labels = labelColumn >= 0 ? new int[rows.size()] : new int[rows.size()];

            for (int r = 0; r < rows.size(); r++) {
                String[] row = rows.get(r);
                int fi = 0;
                for (int c = 0; c < numCols; c++) {
                    if (c == labelColumn) labels[r] = parseLabel(row[c]);
                    else featureData[r][fi++] = parseDouble(row[c]);
                }
            }

            String[] featureNames = new String[featureCols];
            int ni = 0;
            for (int i = 0; i < numCols; i++) if (i != labelColumn) featureNames[ni++] = headers[i];

            Tensor features = Tensor.fromFlat(flatArray(featureData), rows.size(), featureCols);
            return new DataResult(features, labels, featureNames);
        } catch (IOException e) {
            throw new ModelIOException("Failed to read CSV: " + e.getMessage(), e);
        }
    }

    public static DataResult readTsv(String filepath) throws ModelIOException {
        return readCsv(filepath, true, "\t", -1);
    }

    public static DataResult readTsv(String filepath, boolean hasHeader, int labelColumn) throws ModelIOException {
        return readCsv(filepath, hasHeader, "\t", labelColumn);
    }

    // ==================== JSON ====================

    public static DataResult readJson(String filepath) throws ModelIOException {
        try {
            String content;
            try (BufferedReader br = new BufferedReader(new FileReader(filepath, StandardCharsets.UTF_8))) {
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) sb.append(line);
                content = sb.toString().trim();
            }

            List<String[]> rows = new ArrayList<>();
            String[] headers = null;

            if (content.startsWith("[")) {
                String arrayContent = content.substring(1, content.lastIndexOf(']')).trim();
                List<String> objects = splitJsonObjects(arrayContent);
                if (!objects.isEmpty()) {
                    headers = extractJsonKeys(objects.get(0));
                    for (String obj : objects) rows.add(extractJsonValues(obj, headers));
                }
            } else if (content.startsWith("{")) {
                List<String> objects = splitJsonObjects(content);
                if (!objects.isEmpty()) {
                    headers = extractJsonKeys(objects.get(0));
                    for (String obj : objects) rows.add(extractJsonValues(obj, headers));
                }
            }

            if (rows.isEmpty()) throw new ModelIOException("No data found in JSON: " + filepath);

            int numCols = headers.length;
            double[][] featureData = new double[rows.size()][numCols];
            int[] labels = new int[rows.size()];

            for (int r = 0; r < rows.size(); r++) {
                String[] row = rows.get(r);
                for (int c = 0; c < Math.min(row.length, numCols); c++) {
                    featureData[r][c] = parseDouble(row[c]);
                }
            }

            Tensor features = Tensor.fromFlat(flatArray(featureData), rows.size(), numCols);
            return new DataResult(features, labels, headers);
        } catch (IOException e) {
            throw new ModelIOException("Failed to read JSON: " + e.getMessage(), e);
        }
    }

    // ==================== TXT ====================

    public static DataResult readTxt(String filepath) throws ModelIOException {
        return readTxt(filepath, " ");
    }

    public static DataResult readTxt(String filepath, String delimiter) throws ModelIOException {
        try {
            List<double[]> parsedLines = new ArrayList<>();
            List<String> rawLines = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(filepath, StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (!line.trim().isEmpty()) rawLines.add(line.trim());
                }
            }

            if (rawLines.isEmpty()) throw new ModelIOException("No data found in: " + filepath);

            for (String line : rawLines) {
                String[] parts = line.split(delimiter.equals(" ") ? "\\s+" : Pattern.quote(delimiter));
                double[] values = new double[parts.length];
                boolean allNumeric = true;
                for (int i = 0; i < parts.length; i++) {
                    try { values[i] = Double.parseDouble(parts[i]); }
                    catch (NumberFormatException e) { allNumeric = false; break; }
                }
                if (allNumeric) parsedLines.add(values);
            }

            if (parsedLines.isEmpty()) throw new ModelIOException("No numeric data found in: " + filepath);

            int cols = parsedLines.get(0).length;
            double[][] featureData = parsedLines.toArray(new double[0][]);
            String[] headers = new String[cols];
            for (int i = 0; i < cols; i++) headers[i] = "col_" + i;

            Tensor features = Tensor.fromFlat(flatArray(featureData), featureData.length, cols);
            int[] labels = new int[featureData.length];
            return new DataResult(features, labels, headers);
        } catch (IOException e) {
            throw new ModelIOException("Failed to read TXT: " + e.getMessage(), e);
        }
    }

    // ==================== PARQUET ====================

    public static DataResult readParquet(String filepath) throws ModelIOException {
        try {
            org.apache.hadoop.fs.Path hadoopPath = new org.apache.hadoop.fs.Path(filepath);
            org.apache.parquet.hadoop.ParquetReader<org.apache.parquet.example.data.Group> reader =
                org.apache.parquet.hadoop.ParquetReader.builder(
                    new org.apache.parquet.hadoop.example.GroupReadSupport(),
                    hadoopPath
                ).build();

            List<double[]> rows = new ArrayList<>();
            List<String> columnNames = new ArrayList<>();
            org.apache.parquet.example.data.Group group;
            boolean firstGroup = true;

            while ((group = reader.read()) != null) {
                org.apache.parquet.schema.MessageType schema =
                    (org.apache.parquet.schema.MessageType) group.getType();
                int fieldCount = schema.getFieldCount();

                if (firstGroup) {
                    for (int i = 0; i < fieldCount; i++) columnNames.add(schema.getFieldName(i));
                    firstGroup = false;
                }

                double[] row = new double[fieldCount];
                for (int i = 0; i < fieldCount; i++) {
                    org.apache.parquet.schema.Type fieldType = schema.getType(i);
                    try {
                        if (fieldType.isPrimitive()) {
                            switch (fieldType.asPrimitiveType().getPrimitiveTypeName()) {
                                case INT32, INT64 -> row[i] = group.getInteger(i, 0);
                                case FLOAT -> row[i] = group.getFloat(i, 0);
                                case DOUBLE -> row[i] = group.getDouble(i, 0);
                                case BOOLEAN -> row[i] = group.getBoolean(i, 0) ? 1.0 : 0.0;
                                case BINARY, FIXED_LEN_BYTE_ARRAY -> row[i] = parseDouble(group.getString(i, 0));
                                default -> row[i] = 0.0;
                            }
                        }
                    } catch (Exception e) { row[i] = 0.0; }
                }
                rows.add(row);
            }
            reader.close();

            if (rows.isEmpty()) throw new ModelIOException("No data found in Parquet: " + filepath);

            double[][] featureData = rows.toArray(new double[0][]);
            String[] headers = columnNames.toArray(new String[0]);
            Tensor features = Tensor.fromFlat(flatArray(featureData), featureData.length, featureData[0].length);
            int[] labels = new int[featureData.length];
            return new DataResult(features, labels, headers);
        } catch (ModelIOException e) { throw e; }
          catch (Exception e) { throw new ModelIOException("Failed to read Parquet: " + e.getMessage(), e); }
    }

    private static org.apache.parquet.schema.MessageType getParquetSchema(String filepath) throws Exception {
        org.apache.parquet.hadoop.ParquetFileReader fileReader =
            org.apache.parquet.hadoop.ParquetFileReader.open(
                new org.apache.hadoop.conf.Configuration(),
                new org.apache.hadoop.fs.Path(new File(filepath).toURI())
            );
        org.apache.parquet.schema.MessageType schema = fileReader.getFileMetaData().getSchema();
        fileReader.close();
        return schema;
    }

    // ==================== NUMPY ====================

    public static DataResult readNumpy(String filepath) throws ModelIOException {
        try (RandomAccessFile raf = new RandomAccessFile(filepath, "r")) {
            byte[] magic = new byte[6];
            raf.readFully(magic);
            String magicStr = new String(magic, StandardCharsets.US_ASCII);
            if ((magic[0] & 0xFF) != 0x93 || !magicStr.substring(1).startsWith("NUMPY")) throw new ModelIOException("Invalid Numpy file: " + filepath);

            int majorVersion = raf.read();
            int minorVersion = raf.read();

            int headerLen;
            if (majorVersion == 1) {
                headerLen = raf.readUnsignedShort();
            } else {
                headerLen = Integer.reverseBytes(raf.readInt());
            }

            byte[] headerBytes = new byte[headerLen];
            raf.readFully(headerBytes);
            String header = new String(headerBytes, StandardCharsets.US_ASCII).trim();

            String shapeStr = extractNumpyShape(header);
            String dtype = extractNumpyDtype(header);
            boolean fortranOrder = header.contains("True");

            String[] shapeParts = shapeStr.split(",");
            int rows = 1;
            int cols = 1;
            if (shapeParts.length == 1) {
                rows = Integer.parseInt(shapeParts[0].trim());
                cols = 1;
            } else if (shapeParts.length == 2) {
                rows = Integer.parseInt(shapeParts[0].trim());
                cols = Integer.parseInt(shapeParts[1].trim());
            }

            ByteOrder byteOrder = header.contains("<") ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN;

            double[][] data = new double[rows][cols];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    byte[] valBytes = readNumpyValue(raf, dtype);
                    data[r][c] = decodeNumpyValue(valBytes, dtype, byteOrder);
                }
            }

            String[] headers = new String[cols];
            for (int i = 0; i < cols; i++) headers[i] = "col_" + i;

            Tensor features = Tensor.fromFlat(flatArray(data), rows, cols);
            int[] labels = new int[rows];
            return new DataResult(features, labels, headers);
        } catch (IOException e) {
            throw new ModelIOException("Failed to read Numpy: " + e.getMessage(), e);
        }
    }

    private static String extractNumpyShape(String header) {
        int start = header.indexOf("'shape': (");
        if (start < 0) start = header.indexOf("shape=(");
        if (start < 0) return "0";
        start = header.indexOf('(', start);
        int end = header.indexOf(')', start);
        return header.substring(start + 1, end).replace(" ", "");
    }

    private static String extractNumpyDtype(String header) {
        int start = header.indexOf("'descr': '");
        if (start < 0) return "'f8'";
        start += "'descr': '".length();
        int end = header.indexOf("'", start);
        return header.substring(start, end);
    }

    private static byte[] readNumpyValue(RandomAccessFile raf, String dtype) throws IOException {
        int size = getNumpyDtypeSize(dtype);
        byte[] buf = new byte[size];
        raf.readFully(buf);
        return buf;
    }

    private static int getNumpyDtypeSize(String dtype) {
        if (dtype.contains("f4") || dtype.contains("float32") || dtype.contains("i4") || dtype.contains("int32")) return 4;
        if (dtype.contains("f8") || dtype.contains("float64") || dtype.contains("i8") || dtype.contains("int64")) return 8;
        if (dtype.contains("f2") || dtype.contains("float16")) return 2;
        if (dtype.contains("i2") || dtype.contains("int16")) return 2;
        if (dtype.contains("i1") || dtype.contains("int8") || dtype.contains("u1") || dtype.contains("uint8")) return 1;
        return 8;
    }

    private static double decodeNumpyValue(byte[] buf, String dtype, ByteOrder bo) {
        ByteBuffer bb = ByteBuffer.wrap(buf).order(bo);
        if (dtype.contains("f4") || dtype.contains("float32")) return bb.getFloat();
        if (dtype.contains("f8") || dtype.contains("float64")) return bb.getDouble();
        if (dtype.contains("i4") || dtype.contains("int32")) return bb.getInt();
        if (dtype.contains("i8") || dtype.contains("int64")) return bb.getLong();
        if (dtype.contains("i2") || dtype.contains("int16")) return bb.getShort();
        if (dtype.contains("i1") || dtype.contains("int8") || dtype.contains("u1") || dtype.contains("uint8")) return buf[0] & 0xFF;
        return bb.getDouble();
    }

    public static DataResult readNumpyZip(String filepath) throws ModelIOException {
        try {
            java.util.zip.ZipFile zip = new java.util.zip.ZipFile(filepath);
            java.util.Enumeration<? extends java.util.zip.ZipEntry> entries = zip.entries();

            Map<String, Tensor> arrays = new LinkedHashMap<>();
            while (entries.hasMoreElements()) {
                java.util.zip.ZipEntry entry = entries.nextElement();
                if (entry.getName().endsWith(".npy")) {
                    File temp = File.createTempFile("npy_", ".npy");
                    temp.deleteOnExit();
                    try (InputStream is = zip.getInputStream(entry); FileOutputStream fos = new FileOutputStream(temp)) {
                        byte[] buf = new byte[8192];
                        int len;
                        while ((len = is.read(buf)) > 0) fos.write(buf, 0, len);
                    }
                    DataResult dr = readNumpy(temp.getAbsolutePath());
                    arrays.put(entry.getName().replace(".npy", ""), dr.features());
                }
            }
            zip.close();

            if (arrays.isEmpty()) throw new ModelIOException("No .npy files found in: " + filepath);

            String firstKey = arrays.keySet().iterator().next();
            Tensor first = arrays.get(firstKey);
            int rows = first.rows();
            int totalCols = 0;
            for (Tensor t : arrays.values()) totalCols += t.cols();

            double[][] combined = new double[rows][totalCols];
            int offset = 0;
            for (Tensor t : arrays.values()) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < t.cols(); c++) {
                        combined[r][offset + c] = t.get(r, c);
                    }
                }
                offset += t.cols();
            }

            String[] headers = new String[totalCols];
            int hi = 0;
            for (String key : arrays.keySet()) {
                Tensor t = arrays.get(key);
                for (int c = 0; c < t.cols(); c++) headers[hi++] = key + "_" + c;
            }

            Tensor features = Tensor.fromFlat(flatArray(combined), rows, totalCols);
            int[] labels = new int[rows];
            return new DataResult(features, labels, headers);
        } catch (IOException e) {
            throw new ModelIOException("Failed to read NPZ: " + e.getMessage(), e);
        }
    }

    // ==================== SAFETENSORS ====================

    public static DataResult readSafetensors(String filepath) throws ModelIOException {
        try (RandomAccessFile raf = new RandomAccessFile(filepath, "r")) {
            byte[] headerLenBytes = new byte[8];
            raf.readFully(headerLenBytes);
            long headerLen = ByteBuffer.wrap(headerLenBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();

            byte[] headerBytes = new byte[(int) headerLen];
            raf.readFully(headerBytes);
            String header = new String(headerBytes, StandardCharsets.UTF_8);

            Map<String, long[]> tensorInfo = parseSafetensorsHeader(header);

            if (tensorInfo.isEmpty()) throw new ModelIOException("No tensors found in: " + filepath);

            Map<String, double[][]> arrays = new LinkedHashMap<>();
            for (var entry : tensorInfo.entrySet()) {
                String name = entry.getKey();
                long[] info = entry.getValue();
                long offset = info[0];
                long numBytes = info[1];

                raf.seek(offset);
                int numDoubles = (int) (numBytes / 8);
                double[] data = new double[numDoubles];
                for (int i = 0; i < numDoubles; i++) {
                    data[i] = raf.readDouble();
                }

                int rows = data.length;
                double[][] arr = new double[rows][1];
                for (int i = 0; i < rows; i++) arr[i][0] = data[i];
                arrays.put(name, arr);
            }

            int totalRows = arrays.values().iterator().next().length;
            int totalCols = 0;
            for (double[][] arr : arrays.values()) totalCols += arr[0].length;

            double[][] combined = new double[totalRows][totalCols];
            int offset = 0;
            for (double[][] arr : arrays.values()) {
                for (int r = 0; r < totalRows; r++) {
                    System.arraycopy(arr[r], 0, combined[r], offset, arr[r].length);
                }
                offset += arr[0].length;
            }

            String[] headers = new String[totalCols];
            int hi = 0;
            for (String key : arrays.keySet()) {
                for (int c = 0; c < arrays.get(key)[0].length; c++) headers[hi++] = key + "_" + c;
            }

            Tensor features = Tensor.fromFlat(flatArray(combined), totalRows, totalCols);
            int[] labels = new int[totalRows];
            return new DataResult(features, labels, headers);
        } catch (IOException e) {
            throw new ModelIOException("Failed to read Safetensors: " + e.getMessage(), e);
        }
    }

    private static Map<String, long[]> parseSafetensorsHeader(String header) {
        Map<String, long[]> result = new LinkedHashMap<>();
        String[] entries = header.substring(1, header.lastIndexOf('}')).split("\\},\\s*\"");

        for (String entry : entries) {
            int colonIdx = entry.indexOf(':');
            if (colonIdx < 0) continue;
            String name = entry.substring(0, colonIdx).replace("\"", "");
            String meta = entry.substring(colonIdx + 1);

            long[] info = new long[2];
            int startIdx = meta.indexOf("[");
            int endIdx = meta.indexOf("]");
            if (startIdx >= 0 && endIdx > startIdx) {
                String[] parts = meta.substring(startIdx + 1, endIdx).split(",");
                if (parts.length >= 2) {
                    info[0] = Long.parseLong(parts[0].trim());
                    info[1] = Long.parseLong(parts[1].trim());
                }
            }
            result.put(name, info);
        }
        return result;
    }

    // ==================== ARFF ====================

    public static DataResult readArff(String filepath) throws ModelIOException {
        try {
            List<String[]> rows = new ArrayList<>();
            List<String> headers = new ArrayList<>();
            boolean inData = false;

            try (BufferedReader br = new BufferedReader(new FileReader(filepath, StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty() || line.startsWith("%")) continue;

                    if (line.toLowerCase().startsWith("@data")) {
                        inData = true;
                        continue;
                    }

                    if (line.toLowerCase().startsWith("@attribute")) {
                        String name = line.replaceAll("(?i)@attribute\\s+'([^']+)'\\s+.*", "$1");
                        headers.add(name);
                        continue;
                    }

                    if (inData) {
                        String[] parts = line.split(",");
                        rows.add(parts);
                    }
                }
            }

            if (rows.isEmpty()) throw new ModelIOException("No data found in ARFF: " + filepath);

            int numCols = headers.size();
            if (numCols == 0) {
                numCols = rows.get(0).length;
                headers = new ArrayList<>();
                for (int i = 0; i < numCols; i++) headers.add("col_" + i);
            }

            double[][] featureData = new double[rows.size()][numCols];
            int[] labels = new int[rows.size()];

            for (int r = 0; r < rows.size(); r++) {
                String[] row = rows.get(r);
                for (int c = 0; c < Math.min(row.length, numCols); c++) {
                    featureData[r][c] = parseDouble(row[c]);
                }
            }

            String[] headerArr = headers.toArray(new String[0]);
            Tensor features = Tensor.fromFlat(flatArray(featureData), rows.size(), numCols);
            return new DataResult(features, labels, headerArr);
        } catch (IOException e) {
            throw new ModelIOException("Failed to read ARFF: " + e.getMessage(), e);
        }
    }

    // ==================== LIBSVM ====================

    public static DataResult readLibSvm(String filepath) throws ModelIOException {
        try {
            List<double[]> rows = new ArrayList<>();
            List<Integer> labelsList = new ArrayList<>();
            int maxFeatureIdx = 0;

            try (BufferedReader br = new BufferedReader(new FileReader(filepath, StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;

                    String[] parts = line.split("\\s+");
                    labelsList.add(parseLabel(parts[0]));

                    for (int i = 1; i < parts.length; i++) {
                        String[] kv = parts[i].split(":");
                        int idx = Integer.parseInt(kv[0]);
                        if (idx > maxFeatureIdx) maxFeatureIdx = idx;
                    }
                }
            }

            if (rows.isEmpty()) throw new ModelIOException("No data found in LibSVM: " + filepath);

            double[][] featureData = new double[labelsList.size()][maxFeatureIdx];
            int[] labels = new int[labelsList.size()];

            try (BufferedReader br = new BufferedReader(new FileReader(filepath, StandardCharsets.UTF_8))) {
                int r = 0;
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;

                    String[] parts = line.split("\\s+");
                    labels[r] = labelsList.get(r);

                    for (int i = 1; i < parts.length; i++) {
                        String[] kv = parts[i].split(":");
                        int idx = Integer.parseInt(kv[0]) - 1;
                        featureData[r][idx] = Double.parseDouble(kv[1]);
                    }
                    r++;
                }
            }

            String[] headers = new String[maxFeatureIdx];
            for (int i = 0; i < maxFeatureIdx; i++) headers[i] = "feat_" + (i + 1);

            Tensor features = Tensor.fromFlat(flatArray(featureData), labelsList.size(), maxFeatureIdx);
            return new DataResult(features, labels, headers);
        } catch (IOException e) {
            throw new ModelIOException("Failed to read LibSVM: " + e.getMessage(), e);
        }
    }

    // ==================== CONVENIENCE METHODS ====================

    public static Tensor readFeatures(String filepath) throws ModelIOException {
        return read(filepath).features();
    }

    public static int[] readLabels(String filepath) throws ModelIOException {
        return read(filepath).labels();
    }

    public static Tensor readFeaturesCsv(String filepath, int labelColumn) throws ModelIOException {
        return readCsv(filepath, true, ",", labelColumn).features();
    }

    public static int[] readLabelsCsv(String filepath, int labelColumn) throws ModelIOException {
        return readCsv(filepath, true, ",", labelColumn).labels();
    }

    // ==================== HELPERS ====================

    private static String[] parseLine(String line, String delimiter) {
        List<String> parts = new ArrayList<>();
        if (delimiter.equals(",")) {
            boolean inQuote = false;
            StringBuilder current = new StringBuilder();
            for (int i = 0; i < line.length(); i++) {
                char c = line.charAt(i);
                if (c == '"') { inQuote = !inQuote; }
                else if (c == ',' && !inQuote) { parts.add(current.toString().trim()); current = new StringBuilder(); }
                else { current.append(c); }
            }
            parts.add(current.toString().trim());
        } else {
            for (String part : line.split(Pattern.quote(delimiter))) parts.add(part.trim());
        }
        return parts.toArray(new String[0]);
    }

    private static double parseDouble(String s) {
        if (s == null || s.isEmpty()) return 0.0;
        s = s.trim().replace("^\"|\"$|^'|'$", "");
        if (s.equals("?") || s.equals("NA") || s.equals("null") || s.equals("NaN")) return 0.0;
        try { return Double.parseDouble(s); }
        catch (NumberFormatException e) { return (s.hashCode() & 0x7FFFFFFF) % 1000 / 1000.0; }
    }

    private static int parseLabel(String s) {
        if (s == null || s.isEmpty()) return 0;
        s = s.trim().replace("^\"|\"$|^'|'$", "");
        try { return Integer.parseInt(s); }
        catch (NumberFormatException e) { return Math.abs(s.hashCode() % 100); }
    }

    private static double[] flatArray(double[][] arr) {
        int totalLen = 0;
        for (double[] row : arr) totalLen += row.length;
        double[] result = new double[totalLen];
        int offset = 0;
        for (double[] row : arr) { System.arraycopy(row, 0, result, offset, row.length); offset += row.length; }
        return result;
    }

    private static List<String> splitJsonObjects(String arrayContent) {
        List<String> objects = new ArrayList<>();
        int depth = 0;
        int start = -1;
        for (int i = 0; i < arrayContent.length(); i++) {
            char c = arrayContent.charAt(i);
            if (c == '{') { if (depth == 0) start = i; depth++; }
            else if (c == '}') { depth--; if (depth == 0 && start >= 0) { objects.add(arrayContent.substring(start, i + 1)); start = -1; } }
        }
        return objects;
    }

    private static String[] extractJsonKeys(String jsonObject) {
        String content = jsonObject.substring(1, jsonObject.lastIndexOf('}'));
        List<String> keys = new ArrayList<>();
        int i = 0;
        while (i < content.length()) {
            int keyStart = content.indexOf('"', i);
            if (keyStart < 0) break;
            int keyEnd = content.indexOf('"', keyStart + 1);
            if (keyEnd < 0) break;
            keys.add(content.substring(keyStart + 1, keyEnd));
            i = keyEnd + 1;
        }
        return keys.toArray(new String[0]);
    }

    private static String[] extractJsonValues(String jsonObject, String[] keys) {
        String content = jsonObject.substring(1, jsonObject.lastIndexOf('}'));
        String[] values = new String[keys.length];
        for (int k = 0; k < keys.length; k++) {
            String keyPattern = "\"" + keys[k] + "\"";
            int keyIdx = content.indexOf(keyPattern);
            if (keyIdx < 0) { values[k] = "0"; continue; }
            int colonIdx = content.indexOf(':', keyIdx + keyPattern.length());
            if (colonIdx < 0) { values[k] = "0"; continue; }
            int valueStart = colonIdx + 1;
            while (valueStart < content.length() && content.charAt(valueStart) == ' ') valueStart++;
            if (valueStart < content.length() && content.charAt(valueStart) == '"') {
                int valueEnd = content.indexOf('"', valueStart + 1);
                values[k] = valueEnd > valueStart ? content.substring(valueStart + 1, valueEnd) : "0";
            } else {
                int valueEnd = valueStart;
                while (valueEnd < content.length() && content.charAt(valueEnd) != ',' && content.charAt(valueEnd) != '}') valueEnd++;
                values[k] = content.substring(valueStart, valueEnd).trim();
            }
        }
        return values;
    }

    private static int countUnique(int[] arr) {
        Set<Integer> unique = new HashSet<>();
        for (int v : arr) unique.add(v);
        return unique.size();
    }

    public static void printDataInfo(DataResult result) {
        System.out.println("========================================");
        System.out.println("           Data File Info               ");
        System.out.println("========================================");
        System.out.printf("  Format:    %s%n", detectFormat("").name());
        System.out.printf("  Rows:      %d%n", result.features().rows());
        System.out.printf("  Features:  %d%n", result.features().cols());
        System.out.printf("  Labels:    %d unique%n", countUnique(result.labels()));
        System.out.println("----------------------------------------");
        if (result.columnNames() != null && result.columnNames().length > 0) {
            System.out.println("  Columns:");
            for (int i = 0; i < Math.min(result.columnNames().length, 20); i++) {
                System.out.printf("    [%d] %s%n", i, result.columnNames()[i]);
            }
            if (result.columnNames().length > 20) System.out.printf("    ... and %d more%n", result.columnNames().length - 20);
        }
        System.out.println("========================================");
    }
}
