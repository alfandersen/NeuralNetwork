import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @author Alf Andersen
 * @date 2017-10-20
 * Parallel extension class for the basic neural network found in NeuralNetwork.java
 * Task handling is quite expensive, so this implementation will be slower for small networks and faster for larger ones.
 */
public class ParallelNeuralNetwork extends NeuralNetwork {
    ExecutorService executor;
    int processors;

    public ParallelNeuralNetwork(int[] layerSizes, long seed){
        super(layerSizes, seed);
        setupExecutor();
    }

    public ParallelNeuralNetwork(int[] layerSizes){
        super(layerSizes);
        setupExecutor();
    }

    void setupExecutor(){
        processors = Runtime.getRuntime().availableProcessors();
        executor = Executors.newWorkStealingPool();
    }

    @Override
    void feedForward(double[] input) {
        List<Callable<Void>> tasks = new ArrayList<>();

        /*  Initialize input values.
            output of an input unit is its actual input value   */
        for(int nj = 0; nj < neurons[0].length; nj++)
            neurons[0][nj].value = input[nj];


        /*  Move forward through the layers
            for each hidden or output layer unit j:
                compute the net input of unit j with respect to the previous layer, i:      I[j] = ∑ (w[i,j]*O[i]) + θ[j] ;
                compute the output of each unit j:                                          O[j] = sigmoid(I[j]) ;  */

        for(int lj = 1; lj < neurons.length; lj++){
            tasks = new ArrayList<>();
            int li = lj-1;
            int lastTo = 0;
            for(int p = 0; p < processors; p++) {
                final int from = lastTo;
                final int to = p == processors-1 ? neurons[lj].length : (p + 1) * neurons[lj].length / processors;
                lastTo = to;
                final int flj = lj, fli = li;
                tasks.add(() -> {
                    for (int nj = from; nj < to; nj++) {
                        double sum = neurons[flj][nj].bias;
                        for (int ni = 0; ni < neurons[fli].length; ni++) {
                            sum += neurons[fli][ni].value * w[fli][ni][nj];
                        }
                        neurons[flj][nj].value = sigmoid(sum);
                    }
                    return null;
                });
            }
            try {
                executor.invokeAll(tasks);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    void backPropagate(double[] target) {
        List<Callable<Void>> tasks = new ArrayList<>();

        /*  Compute Output Error
            for each unit j in the output layer
                Err[j] = O[j] * ( 1 − O[j] ) * ( T[j] − O[j] ) ;  */
        for(int nj = 0; nj < neurons[outputLayer].length; nj++){
            double deltaOut = target[nj] - neurons[outputLayer][nj].value;
            err[outputLayer][nj] = neurons[outputLayer][nj].value * (1 - neurons[outputLayer][nj].value) * deltaOut;
        }

        /*  Compute Hidden Error
            for each unit j in the hidden layers, from the last to the first hidden layer
                compute the error with respect to the next higher layer, k:         Err[j] = O[j] ( 1 − O[j] ) ∑ (Err[k] * w[j,k]) ;    */
        for(int lk = outputLayer; lk > 0; lk--) {
            tasks.clear();
            int lj = lk-1;
            int lastTo = 0;
            for(int p = 0; p < processors; p++) {
                final int from = lastTo;
                final int to = p == processors - 1 ? neurons[lj].length : (p + 1) * neurons[lj].length / processors;
                lastTo = to;
                final int flk = lk, flj = lj;
                tasks.add(() -> {
                    for (int nj = from; nj < to; nj++) {
                        err[flj][nj] = 0;
                        for (int nk = 0; nk < neurons[flk].length; nk++) {
                            err[flj][nj] += err[flk][nk] * w[flj][nj][nk];
                        }
                        err[flj][nj] *= neurons[flj][nj].value * (1 - neurons[flj][nj].value);
                    }
                    return null;
                });
            }
            try {
                executor.invokeAll(tasks);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        /*  Update Weights and Bias
            for each weight w[i,j] and bias θ[j] in network:
                update weight:      w[i,j] = w[i,j] + leaning_rate * Err[j] * O[i] ;
                update bias:        θ[j] = θ[j] + leaning_rate * Err[j] ;  */
        for(int li = 0; li < outputLayer; li++){
            tasks.clear();
            int lj = li+1;
            int lastTo  = 0;
            for(int p = 0; p < processors; p++) {
                final int from = lastTo;
                final int to = p == processors - 1 ? neurons[lj].length : (p + 1) * neurons[lj].length / processors;
                lastTo = to;
                final int fli = li, flj = lj;
                tasks.add(() -> {
                    for (int nj = from; nj < to; nj++) {
                        for (int ni = 0; ni < neurons[fli].length; ni++) {
                            w[fli][ni][nj] += learningRate * err[flj][nj] * neurons[fli][ni].value;
                        }
                        neurons[flj][nj].bias += learningRate * err[flj][nj];
                    }
                    return null;
                });
            }
            try {
                executor.invokeAll(tasks);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
