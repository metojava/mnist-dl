package ml;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTLoader {

    public static class Dataset {

        private final Tensor images;
        private final int[] labels;

        public Dataset(
                Tensor images,
                int[] labels
        ) {
            this.images = images;
            this.labels = labels;
        }

        public Tensor images() {
            return images;
        }

        public int[] labels() {
            return labels;
        }
    }

    public static Dataset load(
            String imageFile,
            String labelFile
    ) throws IOException {

        try (
                DataInputStream images =
                        new DataInputStream(
                                new FileInputStream(imageFile)
                        );

                DataInputStream labels =
                        new DataInputStream(
                                new FileInputStream(labelFile)
                        )
        ) {

            int imageMagic =
                    images.readInt();

            int numberOfImages =
                    images.readInt();

            int rows =
                    images.readInt();

            int cols =
                    images.readInt();

            int labelMagic =
                    labels.readInt();

            int numberOfLabels =
                    labels.readInt();

            if (imageMagic != 2051) {
                throw new IOException(
                        "Invalid MNIST image file."
                );
            }

            if (labelMagic != 2049) {
                throw new IOException(
                        "Invalid MNIST label file."
                );
            }

            if (numberOfImages != numberOfLabels) {
                throw new IOException(
                        "Number of images and labels differ."
                );
            }

            Tensor tensor =
                    new Tensor(
                            numberOfImages,
                            rows * cols
                    );

            int[] labelArray =
                    new int[numberOfImages];

            for (int i = 0; i < numberOfImages; i++) {

                for (int pixel = 0;
                     pixel < rows * cols;
                     pixel++) {

                    int value =
                            images.readUnsignedByte();

                    /*
                     * Normalize pixels from
                     * [0,255] -> [0,1].
                     */
                    tensor.set(
                            i,
                            pixel,
                            value / 255.0
                    );
                }

                labelArray[i] =
                        labels.readUnsignedByte();
            }

            return new Dataset(
                    tensor,
                    labelArray
            );
        }
    }
}
