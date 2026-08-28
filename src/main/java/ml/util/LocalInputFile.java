package ml.util;

import org.apache.parquet.io.InputFile;
import org.apache.parquet.io.SeekableInputStream;

import java.io.*;
import java.nio.ByteBuffer;

public class LocalInputFile implements InputFile {
    private final File file;
    private final long length;

    public LocalInputFile(File file) {
        this.file = file;
        this.length = file.length();
    }

    public LocalInputFile(String path) {
        this(new File(path));
    }

    @Override
    public long getLength() throws IOException {
        return length;
    }

    @Override
    public SeekableInputStream newStream() throws IOException {
        return new LocalSeekableInputStream(file);
    }

    private static class LocalSeekableInputStream extends SeekableInputStream {
        private final RandomAccessFile raf;
        private long pos;

        public LocalSeekableInputStream(File file) throws FileNotFoundException {
            this.raf = new RandomAccessFile(file, "r");
            this.pos = 0;
        }

        @Override
        public long getPos() {
            return pos;
        }

        @Override
        public void seek(long newPos) {
            this.pos = newPos;
        }

        @Override
        public void readFully(byte[] buffer) throws IOException {
            raf.seek(pos);
            raf.readFully(buffer);
            pos += buffer.length;
        }

        @Override
        public void readFully(byte[] buffer, int start, int len) throws IOException {
            raf.seek(pos);
            raf.readFully(buffer, start, len);
            pos += len;
        }

        @Override
        public void readFully(ByteBuffer buf) throws IOException {
            byte[] b = new byte[buf.remaining()];
            raf.seek(pos);
            raf.readFully(b);
            pos += b.length;
            buf.put(b);
        }

        @Override
        public int read() throws IOException {
            raf.seek(pos);
            int b = raf.read();
            if (b >= 0) pos++;
            return b;
        }

        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            raf.seek(pos);
            int read = raf.read(b, off, len);
            if (read > 0) pos += read;
            return read;
        }

        @Override
        public int read(ByteBuffer buf) throws IOException {
            byte[] b = new byte[buf.remaining()];
            int read = read(b);
            if (read > 0) buf.put(b, 0, read);
            return read;
        }

        @Override
        public void close() throws IOException {
            raf.close();
        }
    }
}
