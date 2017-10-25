import java.util.Arrays;
import java.util.Random;

/**
 * @author Alf Andersen - github.com/alfandersen
 * @date 2017-10-20
 *
 * Basic neural network with back propagation using sigmoid as activation function.
 * Parallel extension class is found in ParallelNeuralNetwork.java
 *
 * The back propagation algorithm is based on pseudo code found in Data Mining 3rd Edition by Han, Kamber & Pei [pp. 401]
 */
public class NeuralNetwork {
    Random random;
    Neuron[][] neurons;
    int outputLayer;
    double[][][] w;
    double learningRate;
    double[][] err;

    /**
     * Constructor which takes an array with layer sizes for the network and a seed for the random initialization of weights and biases.
     * Index 0, of int[] layerSizes, corresponds to the number of units in the input layer.
     * The last index corresponds to the number of units in the output layer.
     * Any elements between 0 and last corresponds to number of units in a hidden layer.
     * The number of elements in int[] layerSizes determines the number of layers in the network.
     *
     * The network is initialized with a default learning rate of 0.1. Use setLearningRate() to change this value.
     * @param layerSizes describing the number of units in each layer.
     * @param seed random seed for the initial weights and biases.
     */
    public NeuralNetwork(int[] layerSizes, long seed){
        if(layerSizes.length < 2)
            throw new IllegalArgumentException(String.format("Not enough layers given. Must be at least two to create Input and Output. You gave %s", Arrays.toString(layerSizes)));
        random = new Random(seed);
        setupLayers(layerSizes);
    }

    /**
     * Constructor which takes an array with layer sizes for the network.
     * Index 0, of int[] layerSizes, corresponds to the number of units in the input layer.
     * The last index corresponds to the number of units in the output layer.
     * Any elements between 0 and last corresponds to number of units in a hidden layer.
     * The number of elements in int[] layerSizes determines the number of layers in the network.
     *
     * The network is initialized with a default learning rate of 0.1. Use setLearningRate() to change this value.
     * @param layerSizes describing the number of units in each layer.
     */
    public NeuralNetwork(int[] layerSizes){
        if(layerSizes.length < 2)
            throw new IllegalArgumentException(String.format("Not enough layers given. Must be at least two to create Input and Output. You gave %s", Arrays.toString(layerSizes)));
        random = new Random();
        setupLayers(layerSizes);
    }

    private void setupLayers(int[] layerSizes){
        learningRate = 0.1;
        neurons = new Neuron[layerSizes.length][];
        w = new double[layerSizes.length-1][][];
        err = new double[neurons.length][];
        outputLayer = neurons.length-1;
        for(int lj = 0; lj < layerSizes.length; lj++){
            int li = lj-1;
            neurons[lj] = new Neuron[layerSizes[lj]];
            err[lj] = new double[layerSizes[lj]];
            if(li >= 0)
                w[li] = new double[layerSizes[li]][layerSizes[lj]];
            for(int nj = 0; nj < neurons[lj].length; nj++) {
                neurons[lj][nj] = new Neuron();
                if(li >= 0){
                    for (int ni = 0; ni < neurons[li].length; ni++) {
                        w[li][ni][nj] = 1-2*random.nextDouble();
                    }
                }
            }
        }
    }

    /**
     * Feeds the input forward through the network and back propagates the error.
     * @param input layer to train on.
     * @param target layer with expected outputs.
     */
    public void train(double[] input, double[] target){
        if(input.length != neurons[0].length)
            throw new IllegalArgumentException("input.length did not match network input layer. Actual: "+input.length+" Expected: "+neurons[0].length);
        if(target.length != neurons[outputLayer].length)
            throw new IllegalArgumentException("target.length did not match network output layer. Actual: "+target.length+" Expected: "+neurons[outputLayer].length);

        feedForward(input);
        backPropagate(target);
    }

    /**
     * Train an entire epoch by giving an array of input sets and their corresponding expected outputs.
     * @param input Two-dimensional array of double values of the form {{input_set_1},{input_set_2},...,{input_set_n}} where n is the last set in the epoch.
     * @param expected Two-dimensional array of double values of the form {{target_set_1},{target_set_2},...,{target_set_n}} where n is the last set in the epoch.
     */
    public void trainEpoch(double[][] input, double[][] expected){
        long time = System.nanoTime();
        String info = "Training epoch ... ";
        System.out.print(info+"  0% complete.");
        for (int i = 0; i < input.length; i++) {
            train(input[i], expected[i]);
            if(System.nanoTime() > time + 1E9) {
                System.out.printf("\r%s%3d%s complete.", info, 100*i/input.length, "%");
                time = System.nanoTime();
            }
        }
        System.out.printf("\r%s%3d%s complete.\n", info, 100, "%");
    }

    /**
     * Feeds an input through the network to predict an output.
     * @param input layer to predict for.
     * @return output layer after the given input has been fed through the network.
     */
    public double[] predict(double[] input){
        if(input.length != neurons[0].length)
            throw new IllegalArgumentException("input.length did not match network input layer. Actual: "+input.length+" Expected: "+neurons[0].length);

        feedForward(input);

        double[] out = new double[neurons[outputLayer].length];
        for(int no = 0; no < out.length; no++){
            out[no] = neurons[outputLayer][no].value;
        }

        return out;
    }

    /**
     * @return the current learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * @param learningRate overwrites the current learning rate.
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    void feedForward(double[] input){
        // Initialize input values.
        for(int nj = 0; nj < neurons[0].length; nj++)
            // output of an input unit is its actual input value
            neurons[0][nj].value = input[nj];


        // Move forward through the layers
        for(int lj = 1; lj < neurons.length; lj++){
            computeLayer(lj);
        }
    }

    void computeLayer(int lj){
        int li = lj-1;
        // for each hidden or output layer unit j:
        for(int nj = 0; nj < neurons[lj].length; nj++){
            // compute the net input of unit j with respect to the previous layer, i:  I[j] = ∑ (w[i,j]*O[i]) + θ[j]
            double sum = neurons[lj][nj].bias;
            for(int ni = 0; ni < neurons[li].length; ni++){
                sum += neurons[li][ni].value * w[li][ni][nj];
            }
            // compute the output of each unit j:   O[j] = sigmoid(I[j]) ;
            neurons[lj][nj].value = sigmoid(sum);
        }
    }

    void backPropagate(double[] target){
        // Compute Output Error for each unit j in the output layer
        for(int nj = 0; nj < neurons[outputLayer].length; nj++){
            // Err[j] = O[j] * ( 1 − O[j] ) * ( T[j] − O[j] ) ;
            double deltaOut = target[nj] - neurons[outputLayer][nj].value;
            err[outputLayer][nj] = neurons[outputLayer][nj].value * (1 - neurons[outputLayer][nj].value) * deltaOut;
        }

        // Compute the hidden error for each layer moving backwards from output layer to input layer
        for(int lj = outputLayer-1; lj >= 0; lj--) {
            computeHiddenError(lj);
        }

        // Update Weights and Bias for each weight w[i,j] and bias θ[j] in network:
        for(int lj = 1; lj <= outputLayer; lj++){
            updateWeightsAndBiases(lj);
        }
    }

    void computeHiddenError(int lj){
        int lk = lj+1;
        // Compute Hidden Error for each unit j in the hidden layer lj
        for (int nj = 0; nj < neurons[lj].length; nj++) {
            // Err[j] = O[j] ( 1 − O[j] ) ∑ (Err[k] * w[j,k]) ;
            err[lj][nj] = 0;
            for(int nk = 0; nk < neurons[lk].length; nk++) {
                err[lj][nj] += err[lk][nk] * w[lj][nj][nk];
            }
            err[lj][nj] *= neurons[lj][nj].value * (1 - neurons[lj][nj].value);
        }
    }

    void updateWeightsAndBiases(int lj){
        int li = lj-1;
        // Update Weights and Bias for each weight w[i,j] and bias θ[j]
        for(int nj = 0; nj < neurons[lj].length; nj++){
            for(int ni = 0; ni < neurons[li].length; ni++){
                // update weight:  w[i,j] = w[i,j] + leaning_rate * Err[j] * O[i] ;
                w[li][ni][nj] += learningRate * err[lj][nj] * neurons[li][ni].value;
            }
            // update bias:  θ[j] = θ[j] + leaning_rate * Err[j] ;
            neurons[lj][nj].bias += learningRate * err[lj][nj];
        }
    }

    static double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    class Neuron {
        double value;
        double bias;

        Neuron(){
            bias = 1-(2*random.nextDouble());
        }
    }
}
