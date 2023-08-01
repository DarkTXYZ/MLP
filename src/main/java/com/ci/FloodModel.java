package com.ci;

import com.ci.loss_function.LossFunction;
import com.ci.loss_function.MAE;
import com.ci.loss_function.MSE;
import org.apache.commons.collections4.ListUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.shade.guava.math.IntMath;

import java.io.BufferedWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import java.io.FileWriter;

public class FloodModel {
    public static void main(String[] args) {
        
        try {
//            FileWriter fw = new FileWriter("FloodModelLog.txt", true);
//            BufferedWriter bw = new BufferedWriter(fw);
            
            List<DataSet> dataset = Util.getDataSet("csv/flood.csv", ",", 0, 1, 8, 8);
            
            int partitionSize = IntMath.divide(dataset.size(), 10, RoundingMode.UP);
            List<List<DataSet>> folds = ListUtils.partition(dataset, partitionSize);
            
            Double mean = 340.3040;
            Double std = 120.4517;
            
            int index = 0;
            
            String modelStructure = "8-1T";

//            Double[] lr = {0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0};
            Double[] lr = {0.003, 0.01, 0.03, 0.1, 0.3};
//            Double[] lr = {0.1};
//            Double[] mr = {0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0};
            Double[] mr = {0.003, 0.01, 0.03, 0.1, 0.3};
//            Double[] mr = {0.3};
            List<Double> lrs = new ArrayList<Double>(Arrays.asList(lr));
            List<Double> mrs = new ArrayList<Double>(Arrays.asList(mr));
            List<Double> avgError = new ArrayList<>();
            List<Double> avgAccuracy = new ArrayList<>();
            LossFunction lossFunction = new MSE();
            
            for (Double learningRate : lrs) {
                for (Double momentumRate : mrs) {
                    List<Double> error = new ArrayList<>();
//
//                    bw.write(modelStructure + " " + "Learning Rate: " + learningRate +
//                        ", Momentum Rate: " +
//                        momentumRate);
//                    bw.newLine();
                    
                    for (int testFold = 0; testFold < 10; ++testFold) {
                        
                        MLP model =
                            new MLP(modelStructure, lossFunction, learningRate, momentumRate);
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
                    }
                    Collections.sort(error);
                    avgError.add(error.stream().mapToDouble(f -> f).sum() / 10.0);
//                    bw.write("Average " + lossFunction + " Error: " +
//                        error.stream().mapToDouble(f -> f).sum() / 10.0);
//                    bw.newLine();
                }
            }
//            bw.close();
            
            Util.writeArray(avgError,null,"FloodModelLog2.txt",modelStructure);
            
            
            
        } catch (Exception e) {
            System.out.println(e);
        }
        
        
    }
}
