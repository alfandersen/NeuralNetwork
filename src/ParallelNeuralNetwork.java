import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @author Alf Andersen
 * @date 2017-10-20
 * Parallel extension class for the basic neural network found in NeuralNetwork.java
 * Task handling is quite expensive, so this implementation optimizes to only run in parallel when it can run faster by doing so.
 * Note: The constant task creation and execution might over time lead to a slower running time than sequential.
 */
public class ParallelNeuralNetwork extends NeuralNetwork {
    ExecutorService executor;
    List<Callable<Void>> tasks;
    boolean[] doParallel;
    long[] parTimes;
    int processors;

    public ParallelNeuralNetwork(int[] layerSizes, long seed){
        super(layerSizes, seed);
        setupExecutor();
        analyzeLayers();
    }

    public ParallelNeuralNetwork(int[] layerSizes){
        super(layerSizes);
        setupExecutor();
        analyzeLayers();
    }

    void setupExecutor(){
        processors = Runtime.getRuntime().availableProcessors();
        executor = Executors.newWorkStealingPool();
        tasks = new ArrayList<>();
    }

    void analyzeLayers() {
        doParallel = new boolean[w.length];

        for (int lj = 0; lj < doParallel.length; lj++) {
            long parTime = 0;
            long seqTime = 0;
            for(int i = 0; i < 100; i++) {
                long start = System.nanoTime();
                super.computeLayer(lj+1);
                if(i > 10) seqTime += System.nanoTime()-start;
                start = System.nanoTime();
                computeLayerParallel(lj+1);
                if(i > 10) parTime += System.nanoTime()-start;
            }
            doParallel[lj] = parTime < seqTime;
//            System.out.printf("%.3f < %.3f\n", 1E-6*parTime, 1E-6*seqTime);
        }
    }

    @Override
    void computeLayer(int lj) {
        if(doParallel[lj-1]) computeLayerParallel(lj);
        else super.computeLayer(lj);
    }

    @Override
    void computeHiddenError(int lj) {
        if(doParallel[lj]) computeHiddenErrorParallel(lj);
        else super.computeHiddenError(lj);
    }

    @Override
    void updateWeightsAndBiases(int lj) {
        if(doParallel[lj-1]) updateWeightsAndBiasesParallel(lj);
        else super.updateWeightsAndBiases(lj);
    }

    private void computeLayerParallel(int lj) {
        tasks.clear();
        int li = lj-1;
        int lastTo = 0;
        // Create a task for each available processor
        for(int p = 0; p < processors; p++) {
            // divide the number of units computed by each task evenly between the number of tasks
            final int from = lastTo;
            final int to = p == processors-1 ? neurons[lj].length : (p + 1) * neurons[lj].length / processors;
            lastTo = to;
            final int flj = lj, fli = li;
            tasks.add(() -> {
                // for each hidden or output layer unit j:
                for (int nj = from; nj < to; nj++) {
                    // compute the net input of unit j with respect to the previous layer, i:  I[j] = ∑ (w[i,j]*O[i]) + θ[j]
                    double sum = neurons[flj][nj].bias;
                    for (int ni = 0; ni < neurons[fli].length; ni++) {
                        sum += neurons[fli][ni].value * w[fli][ni][nj];
                    }
                    // compute the output of each unit j:   O[j] = sigmoid(I[j]) ;
                    neurons[flj][nj].value = sigmoid(sum);
                }
                return null;
            });
        }
        try {
            // Execute all tasks
            executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        tasks.clear();
    }

    void computeHiddenErrorParallel(int lj) {
        tasks.clear();
        int lk = lj+1;
        int lastTo = 0;
        // Create a task for each available processor
        for(int p = 0; p < processors; p++) {
            // divide the number of units computed by each task evenly between the number of tasks
            final int from = lastTo;
            final int to = p == processors - 1 ? neurons[lj].length : (p + 1) * neurons[lj].length / processors;
            lastTo = to;
            final int flk = lk, flj = lj;
            tasks.add(() -> {
                // Compute Hidden Error for each unit j in the hidden layer lj
                for (int nj = from; nj < to; nj++) {
                    // Err[j] = O[j] ( 1 − O[j] ) ∑ (Err[k] * w[j,k]) ;
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
        tasks.clear();
    }

    void updateWeightsAndBiasesParallel(int lj) {
        tasks.clear();
        int li = lj-1;
        int lastTo  = 0;
        // Create a task for each available processor
        for(int p = 0; p < processors; p++) {
            // divide the number of units computed by each task evenly between the number of tasks
            final int from = lastTo;
            final int to = p == processors - 1 ? neurons[lj].length : (p + 1) * neurons[lj].length / processors;
            lastTo = to;
            final int fli = li, flj = lj;
            tasks.add(() -> {
                // Update Weights and Bias for each weight w[i,j] and bias θ[j]
                for (int nj = from; nj < to; nj++) {
                    for (int ni = 0; ni < neurons[fli].length; ni++) {
                        // update weight:  w[i,j] = w[i,j] + leaning_rate * Err[j] * O[i] ;
                        w[fli][ni][nj] += learningRate * err[flj][nj] * neurons[fli][ni].value;
                    }
                    // update bias:  θ[j] = θ[j] + leaning_rate * Err[j] ;
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
        tasks.clear();
    }

    public boolean[] getParallelization() {
        return doParallel;
    }
}
