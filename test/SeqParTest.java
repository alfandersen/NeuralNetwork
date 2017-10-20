import java.util.Arrays;

/*
NOTE:
ParallelNeuralNetwork uses computeLayerParallel(int) and computeLayer(int) in analyzeLayers()
to determine parallel layers. Therefore an untrained parallel network does not have the same
unit values as an untrained sequential network. These values are overwritten as soon as any other
action is taken.
 */
public class SeqParTest {
    // XOR
    static double[][] input =    {{0,0}, {0,1}, {1,0}, {1,1}};
    static double[][] expected = { {0},   {1},   {1},   {0}};

    static int[] layers = {input[0].length, 33, 42, expected[0].length};


    public static void main(String[] args) {

        NeuralNetwork snn = new NeuralNetwork(layers, 20102017L);
        ParallelNeuralNetwork pnn = new ParallelNeuralNetwork(layers, 20102017L);

        // force parallel computation
        pnn.doParallel = new boolean[] {true, true, true};

        int epochs = 1024;
        if(testEpochs(snn, pnn, epochs))
            System.out.println("Parallel and Sequential networks behaved equally with equal values throughout "
                    +epochs+" epochs of training and testing.");

        snn = new NeuralNetwork(layers, 20102017L);
        pnn = new ParallelNeuralNetwork(layers, 20102017L);
        System.out.println(testPredict(snn, pnn) ?
                "Predicted networks are equal for an untrained network." :
                "FAIL!! Predicted networks are NOT equal for an untrained network.");
    }

    static boolean testEpochs(NeuralNetwork a, NeuralNetwork b, int epochs){
        for(int epoch = 1; epoch <= epochs; epoch ++){
            if(!(testTrain(a, b) && testPredict(a,b))){
                System.out.println("FAIL on epoch "+epoch);
                return false;
            }
        }
        return true;
    }

    static boolean testPredict(NeuralNetwork a, NeuralNetwork b){
        for(int set = 0; set < input.length; set++){
            a.predict(input[set]);
            b.predict(input[set]);
            if(!testEquality(a, b)) {
                System.out.println("FAIL in prediction on set "+set);
                return false;
            }
        }
        return true;
    }

    static boolean testTrain(NeuralNetwork a, NeuralNetwork b){
        for(int set = 0; set < layers[0]; set++){
            a.train(input[set], expected[set]);
            b.train(input[set], expected[set]);
            if(!testEquality(a, b)) {
                System.out.println("FAIL in Training on set "+set);
                return false;
            }
        }
        return true;
    }

    static boolean testEquality(NeuralNetwork a, NeuralNetwork b){
        for(int lj = 0; lj < layers.length; lj++){
            for(int nj = 0; nj < layers[lj]-1; nj++){
                if(a.neurons[lj][nj].value != b.neurons[lj][nj].value) {
                    System.out.println("FAIL on neuron "+lj+","+nj + "   "+ a.neurons[lj][nj].value +" != "+ b.neurons[lj][nj].value);
                    return false;
                }
                if(a.neurons[lj][nj].bias != b.neurons[lj][nj].bias) {
                    System.out.println("FAIL on bias "+lj+","+nj + "   "+ a.neurons[lj][nj].bias +" != "+ b.neurons[lj][nj].bias);
                    return false;
                }
                if(!Arrays.equals(a.w[lj][nj], b.w[lj][nj])) {
                    System.out.println("FAIL on weight "+lj+","+nj + "   "+ Arrays.toString(a.w[lj][nj]) + " != "+ Arrays.toString(b.w[lj][nj]));
                    return false;
                }
            }
        }
        return true;
    }
}
