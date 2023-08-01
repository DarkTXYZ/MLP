package com.ci;

import com.ci.loss_function.BCE;
import com.ci.loss_function.MAE;
import org.apache.commons.collections4.ListUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.math.IntMath;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class CrossPatModel {
    public static void main(String[] args) throws IOException, InterruptedException {

        
        List<DataSet> dataset = Util.getDataSet("csv/cross_pat_one_label.csv", ",", 0, 1, 2, 2);
        
        int partitionSize = IntMath.divide(dataset.size(), 10, RoundingMode.UP);
        List<List<DataSet>> folds = ListUtils.partition(dataset, partitionSize);
        
        Double mean = 0.0;
        Double std = 1.0;
        
        int index = 0;
        
        String modelStructure = "2-5T-3S-4T-1S";

        Double[] lr = {0.003, 0.01, 0.03, 0.1, 0.3};
        Double[] mr = {0.003, 0.01, 0.03, 0.1, 0.3};
        
        List<Double> lrs = new ArrayList<Double>(Arrays.asList(lr));
        List<Double> mrs = new ArrayList<Double>(Arrays.asList(mr));
        List<Double> avgError = new ArrayList<>();
        List<Double> avgAccuracy = new ArrayList<>();

        INDArray confusion_matrix = Nd4j.zeros(4);
        
        for (Double learningRate : lrs) {
            for (Double momentumRate : mrs) {
                List<Double> error = new ArrayList<>();
                List<Double> accuracy = new ArrayList<>();
                
                for (int testFold = 0; testFold < 10; ++testFold) {
                    
                    ClassificationMLP model =
                        new ClassificationMLP(modelStructure, new BCE(), learningRate,
                            momentumRate);
                    model.setMean(mean);
                    model.setStd(std);
                    
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
                    model.train(trainData);
                    model.test(testData);
                    
                    error.add(model.getAverageError());
                    accuracy.add(model.getAccuracy());
                }
                avgError.add(error.stream().mapToDouble(f -> f).sum() / 10.0);
                avgAccuracy.add(accuracy.stream().mapToDouble(f -> f).sum() / 10.0);
            }
        }
        
        Util.writeArray(avgError,avgAccuracy,"CrossPatLog.txt",modelStructure);
    }
}
