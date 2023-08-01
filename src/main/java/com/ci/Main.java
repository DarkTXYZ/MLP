package com.ci;

import com.ci.activation_function.IdentityFunction;
import com.ci.activation_function.TanhFunction;
import com.ci.activation_function.UnitStepFunction;
import com.ci.loss_function.LossFunction;
import com.ci.loss_function.MAE;
import org.apache.commons.collections4.ListUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.shade.guava.math.IntMath;

import javax.xml.crypto.Data;
import java.io.File;
import java.io.IOException;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

public class Main {
    public static void main(String[] args) throws IOException, InterruptedException {
        
        LossFunction l = new MAE();
        
        INDArray a = Nd4j.rand(2);
        INDArray b = Nd4j.ones(1);
        
        System.out.println(a.logEntropy());
        System.out.println(b);
        System.out.println(l.error(a,b));
        System.out.println(l.errorDiff(a,b));
        
//        List<DataSet> dataset = Util.getDataSet("csv/xor.csv", ",", 0, 1,2,2);
//
//        int partitionSize = IntMath.divide(dataset.size(), 10, RoundingMode.UP);
//        List<List<DataSet>> folds = ListUtils.partition(dataset, partitionSize);
//
//        int index = 0;
//        for (int testFold = 0; testFold < 10; ++testFold) {
//            MLP model = new MLP("2-2R-1R", 0.1, 0.1);
//
//            List<List<DataSet>> temp = new ArrayList<>();
//
//            for (int trainFold = 0; trainFold < 10; ++trainFold) {
//                if (testFold == trainFold)
//                    continue;
//                temp.add(folds.get(trainFold));
//            }
//
//            List<DataSet> trainData = temp.stream()
//                .flatMap(Collection::stream)
//                .collect(Collectors.toList());
//            List<DataSet> testData = folds.get(testFold);
//
//            System.out.println("Fold " + (testFold + 1));
//            model.train(trainData);
//            model.test(testData);
//
//        }


//        for(int testBatch = 0 ; testBatch < 10 ; ++testBatch) {
//            int index = 0;
//            Double avgError = 0.0;
//            double trainSize = 0.0;
//            MLP model = new MLP("2-2S-2R-1R" , 0.02);
//
//            for(int i = 0 ; i < 10 ; ++i) {
//                if(testBatch == i)
//                    continue;
//
////                System.out.println((i+1) + "'s batch");
//                DataSet batch = dataset.get(i);
//
//                model.train(batch);
////                break;
////                System.out.println(model);
//
//                trainSize += batch.numExamples();
//                avgError += model.getSumError();
//            }
//            avgError /= trainSize;
//            System.out.println("Epoch " + testBatch + ": " + avgError);
//
//        }

//        while(train.hasNext()) {
//            DataSet cur = train.next();
//            System.out.println(cur.asList());
//        }
//        Scanner scanner = new Scanner(System.in);
//        System.out.println("Enter your Neural Network:");
//        String neuralNetwork = scanner.nextLine();
//        MLP model = new MLP("8-4S-2S-1S" , 0.03);
//        int index = 0;
//
//        model.train(records);

//        System.out.println(model);
    }
}