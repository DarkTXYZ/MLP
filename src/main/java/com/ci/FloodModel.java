package com.ci;

import com.ci.loss_function.LossFunction;
import com.ci.loss_function.MSE;
import org.apache.commons.collections4.ListUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.shade.guava.math.IntMath;

import java.io.IOException;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class FloodModel {
    public static void main(String[] args) throws IOException, InterruptedException {
        List<DataSet> dataset = Util.getDataSet("csv/flood.csv", ",", 0, 1, 8, 8);
        
        int partitionSize = IntMath.divide(dataset.size(), 10, RoundingMode.UP);
        List<List<DataSet>> folds = ListUtils.partition(dataset, partitionSize);
        
        Double mean = 340.3040;
        Double std = 120.4517;
        
        List<String> modelStructure = new ArrayList<>();
        modelStructure.add("8-1T");
        modelStructure.add("8-8T-2T-1T");
        modelStructure.add("8-8T-6T-2T-1T");
        modelStructure.add("8-16T-1T");
        modelStructure.add("8-16T-8T-2T-1T");

        Double[] lr = {0.003, 0.01, 0.03, 0.1, 0.3};
        Double[] mr = {0.003, 0.01, 0.03, 0.1, 0.3};
        List<Double> lrs = new ArrayList<Double>(Arrays.asList(lr));
        List<Double> mrs = new ArrayList<Double>(Arrays.asList(mr));

        for (String model : modelStructure) {
            List<Double> avgError = new ArrayList<>();
            List<Double> avgAccuracy = new ArrayList<>();
            LossFunction lossFunction = new MSE();

            for (Double learningRate : lrs) {
                for (Double momentumRate : mrs) {
                    List<Double> error = new ArrayList<>();

                    for (int testFold = 0; testFold < 10; ++testFold) {

                        MLP trainingModel =
                            new MLP(model, lossFunction, learningRate, momentumRate);
                        trainingModel.setMean(mean);
                        trainingModel.setStd(std);

                        List<List<DataSet>> temp = new ArrayList<>();

                        for (int trainFold = 0; trainFold < 10; ++trainFold) {
                            if (testFold == trainFold)
                                continue;
                            temp.add(folds.get(trainFold));
                        }

                        List<DataSet> trainData = temp.stream()
                            .flatMap(Collection::stream)
                            .collect(Collectors.toList());
                        List<DataSet> testData = folds.get(testFold);

                        System.out.println("Fold " + (testFold + 1));
                        trainingModel.train(trainData);
                        trainingModel.test(testData);

                        error.add(trainingModel.getAverageError());
                    }
                    avgError.add(error.stream().mapToDouble(f -> f).sum() / 10.0);
                }
            }
            Util.writeArray(avgError, null, "FloodModelLog2.txt", model);
            break;

        }
    }
}
