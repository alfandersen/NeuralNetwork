import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

// Adaptive decreasing learning rate is calculated by: learningrate = 0.5/(1+epoch)+0.01
// I haven't played too much with different learning rates and layer structures.
//
/* Some results:

LAYER STRUCTURE: [784, 56, 20, 10]
LEARNING RATE =  0.5/(1+epoch)+0.01
RANDOM SEED:     1234
Epoch:   1	Test Error Rate:   6.99 %
Epoch:   5	Test Error Rate:   4.00 %
Epoch:  10	Test Error Rate:   3.68 %
Epoch:  15	Test Error Rate:   3.72 %
Epoch:  20	Test Error Rate:   3.69 %

LAYER STRUCTURE: [784, 112, 56, 10]
LEARNING RATE =  0.5/(1+epoch)+0.01
RANDOM SEED:     1234
Epoch:   1	Test Error Rate:   6.11 %
Epoch:   5	Test Error Rate:   3.67 %
Epoch:  10	Test Error Rate:   3.51 %
Epoch:  15	Test Error Rate:   3.36 %
Epoch:  20	Test Error Rate:   3.29 %
Epoch:  25	Test Error Rate:   3.19 %
Epoch:  30	Test Error Rate:   3.14 %
Epoch:  35	Test Error Rate:   3.16 %

LAYER STRUCTURE: [784, 300, 30, 10]
LEARNING RATE =  0.5/(1+epoch)+0.01
RANDOM SEED:     1234
Epoch:   1	Test Error Rate:   6.12 %
Epoch:   5	Test Error Rate:   4.02 %
Epoch:  10	Test Error Rate:   3.66 %
Epoch:  15	Test Error Rate:   3.34 %
Epoch:  20	Test Error Rate:   3.28 %
Epoch:  25	Test Error Rate:   3.25 %
Epoch:  30	Test Error Rate:   3.32 %
Epoch:  35	Test Error Rate:   3.29 %
Epoch:  40	Test Error Rate:   3.20 %
Epoch:  45	Test Error Rate:   3.19 %
Epoch:  50	Test Error Rate:   3.17 %
Training time: 47 minutes with parallel implementation.

LAYER STRUCTURE: [784, 112, 112, 10]
LEARNING RATE =  0.5/(1+epoch)+0.01
RANDOM SEED:     1234
Epoch:   1	Test Error Rate:   6.46 %
Epoch:   5	Test Error Rate:   3.84 %
Epoch:  10	Test Error Rate:   3.29 %
Epoch:  15	Test Error Rate:   3.21 %
Epoch:  20	Test Error Rate:   3.14 %
Epoch:  25	Test Error Rate:   3.16 %
Epoch:  30	Test Error Rate:   3.21 %
Epoch:  35	Test Error Rate:   3.16 %
Epoch:  40	Test Error Rate:   3.06 %
Epoch:  45	Test Error Rate:   3.07 %
Epoch:  50	Test Error Rate:   3.06 %
Epoch:  55	Test Error Rate:   3.09 %
Training time: 35 minutes with sequential implementation.

*/
class MNISTTest {
    public static void main(String[] args) throws IOException {
        // Hard coded resource root for the win!
        MNIST dataset = new MNIST("./resources/");

        int[] layerSizes = {MNIST.pixelsPerImage, 150,20, MNIST.labelAmount};
        long randomSeed = 1234L;

        // My 4 core cpu seems to starts benefiting from a parallel network with a hidden layer of around 100 units
        // accumulating to a total of 78,400 weights between the input and first hidden layer.
//        NeuralNetwork nn = new NeuralNetwork(layerSizes, randomSeed);
        ParallelNeuralNetwork nn = new ParallelNeuralNetwork(layerSizes, randomSeed);

        System.out.printf("LAYER STRUCTURE: %s\n", Arrays.toString(layerSizes));
        System.out.printf("LEARNING RATE =  %s\n", "0.5/(1+epoch)+0.01");
        System.out.printf("RANDOM SEED:     %d\n", randomSeed);
        System.out.printf("Parallelization: %s\n", Arrays.toString(nn.getParallelization()));

        int testFrequency = 5;
        int printFrequency = 2000;

        for(int epoch = 1; epoch <= 100; epoch++) {
            nn.setLearningRate(0.5/(1+epoch)+0.01);
            long time = System.nanoTime();
            for (int i = 0; i < MNIST.trainAmount; i++) {
                double[][] trainingPair = dataset.getTrainingPair(i);
                nn.train(trainingPair[0], trainingPair[1]);
                if(i % printFrequency == 0) {
                    double avgTime = (1E-9*(System.nanoTime()-time))/printFrequency;
                    long estmTime = (long)(MNIST.trainAmount*avgTime);
                    System.out.printf("\rEpoch: %3s\tSet: %5s\tEstimated epoch time: %d:%02d", epoch, i, estmTime/60, estmTime%60);
                    time = System.nanoTime();
                }
            }

            if(epoch == 1 || epoch % testFrequency == 0) {
                int correct = 0;
                System.out.printf("\rEpoch: %3s\tTesting ...", epoch);
                for (int i = 0; i < MNIST.testAmount; i++) {
                    double[][] testPair = dataset.getTestPair(i);
                    double[] prediction = nn.predict(testPair[0]);
                    if (maxIndex(prediction) == maxIndex(testPair[1]))
                        correct++;
                }
                System.out.printf("\rEpoch: %3s\tTest Error Rate: %8s\n", epoch, String.format("%.2f %s", (100.*(MNIST.testAmount - correct))/ MNIST.testAmount, "%"));
            }
        }
    }

    static int maxIndex(double[] prediction){
        int max = 0;
        for(int i = 1; i < prediction.length; i++)
            max = prediction[i] > prediction[max] ? i : max;
        return max;
    }
}

/*

Data sets and description from source: http://yann.lecun.com/exdb/mnist/


FILE FORMATS FOR THE MNIST DATABASE
The data is stored in a very simple file format designed for storing vectors and multidimensional matrices. General info on this format is given at the end of this page, but you don't need to read that to use the data files.

All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and other low-endian machines must flip the bytes of the header.

There are 4 files:

train-images-idx3-ubyte: training set images
train-labels-idx1-ubyte: training set labels
t10k-images-idx3-ubyte:  test set images
t10k-labels-idx1-ubyte:  test set labels

The training set contains 60000 examples, and the test set 10000 examples.

The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.
TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
 */
public class MNIST {
    private String path;
    private static final String trainImageFile = "train-images.idx3-ubyte";
    private static final String trainLabelFile = "train-labels.idx1-ubyte";
    private static final String testImageFile = "t10k-images.idx3-ubyte";
    private static final String testLabelFile = "t10k-labels.idx1-ubyte";
    public static final int trainAmount = 60_000;
    public static final int testAmount  = 10_000;
    public static final int pixelsPerImage = 784;
    public static final int labelAmount =     10;
    private byte[] pixels;
    private byte[] label;
    private double[][][] trainPairs;
    private double[][][] testPairs;

    public MNIST(String path) throws IOException {
        this.path = path;
        pixels = new byte[pixelsPerImage];
        label = new byte[1];
        trainPairs = loadDataPairs(trainImageFile, trainLabelFile, trainAmount);
        testPairs = loadDataPairs(testImageFile, testLabelFile, testAmount);
    }

    public double[][] getTrainingPair(int index) throws IOException {
        return trainPairs[index];
    }

    public double[][] getTestPair(int index) throws IOException {
        return testPairs[index];
    }

    private double[][][] loadDataPairs(String imageFile, String labelFile, int amount) throws IOException {
        FileInputStream imageReader = new FileInputStream(path+ imageFile);
        FileInputStream labelReader = new FileInputStream(path+ labelFile);
        imageReader.skip(16);
        labelReader.skip(8);
        double[][][] pairs = new double[amount][2][];
        for(int i = 0; i < amount; i++) {
            pairs[i][0] = loadImage(imageReader);
            pairs[i][1] = loadLabel(labelReader);
        }
        imageReader.close();
        labelReader.close();
        return pairs;
    }

    private double[] loadImage(FileInputStream stream) throws IOException {
        double[] output = new double[pixelsPerImage];
        stream.read(pixels);
        for(int i = 0; i < pixelsPerImage; i++)
            output[i] = (pixels[i] & 0xff) * 1./255;
        return output;
    }

    private double[] loadLabel(FileInputStream stream) throws IOException {
        double[] output = new double[labelAmount];
        output[stream.read()] = 1;
        return output;
    }
}
