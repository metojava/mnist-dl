package ml;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTLoader {

    private MNISTLoader() {
    }

    public static Dataset load(
            String imageFile,
            String labelFile
    ) throws IOException {

        long start = System.nanoTime();

        System.out.println(
                "Loading images: " + imageFile
        );

        System.out.println(
                "Loading labels: " + labelFile
        );

        Tensor images =
                loadImages(imageFile);

        int[] labels =
                loadLabels(labelFile);

        if (images.rows() != labels.length) {
            throw new IOException(
                    "Image count (" +
                    images.rows() +
                    ") does not match label count (" +
                    labels.length +
                    ")."
            );
        }

        long end = System.nanoTime();

        double seconds =
                (end - start) / 1_000_000_000.0;

        System.out.printf(
                "Loaded %d images in %.2f seconds%n",
                images.rows(),
                seconds
        );

        return new Dataset(
                images,
                labels
        );
    }

    private static Tensor loadImages(
            String filename
    ) throws IOException {

        try (DataInputStream in =
                     new DataInputStream(
                             new BufferedInputStream(
                                     new FileInputStream(
                                             filename
                                     ),
                                     1024 * 1024
                             )
                     )) {

            // IDX magic number
            int magic = in.readInt();

            if (magic != 2051) {
                throw new IOException(
                        "Invalid MNIST image file. " +
                        "Expected magic number 2051, got " +
                        magic
                );
            }

            int numberOfImages =
                    in.readInt();

            int rows =
                    in.readInt();

            int columns =
                    in.readInt();

            System.out.println(
                    "Images: " +
                    numberOfImages +
                    " (" +
                    rows +
                    "x" +
                    columns +
                    ")"
            );

            int inputSize =
                    rows * columns;

            Tensor tensor =
                    new Tensor(
                            numberOfImages,
                            inputSize
                    );

            /*
             * MNIST stores pixels as unsigned bytes.
             *
             * We normalize them to:
             *
             * 0.0 -> black
             * 1.0 -> white
             */
            for (int i = 0;
                 i < numberOfImages;
                 i++) {

                for (int j = 0;
                     j < inputSize;
                     j++) {

                    int pixel =
                            in.readUnsignedByte();

                    tensor.set(
                            i,
                            j,
                            pixel / 255.0
                    );
                }
            }

            return tensor;
        }
    }

    private static int[] loadLabels(
            String filename
    ) throws IOException {

        try (DataInputStream in =
                     new DataInputStream(
                             new BufferedInputStream(
                                     new FileInputStream(
                                             filename
                                     ),
                                     1024 * 1024
                             )
                     )) {

            int magic = in.readInt();

            if (magic != 2049) {
                throw new IOException(
                        "Invalid MNIST label file. " +
                        "Expected magic number 2049, got " +
                        magic
                );
            }

            int numberOfLabels =
                    in.readInt();

            System.out.println(
                    "Labels: " +
                    numberOfLabels
            );

            int[] labels =
                    new int[numberOfLabels];

            for (int i = 0;
                 i < numberOfLabels;
                 i++) {

                labels[i] =
                        in.readUnsignedByte();
            }

            return labels;
        }
    }

    public record Dataset(
            Tensor images,
            int[] labels
    ) {
    }
}
