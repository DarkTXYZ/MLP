package com.ci;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.guava.collect.Lists;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Util {
    
    public static INDArray listToINDArray(List<Double> input) {
        double[] arr = input.stream().mapToDouble(Double::doubleValue).toArray();
        return Nd4j.create(arr);
    }
    
    public static List<List<String>> readCSV(String path) {
        List<List<String>> records = new ArrayList<List<String>>();
        try (CSVReader csvReader = new CSVReader(new FileReader(path))) {
            String[] values = null;
            while ((values = csvReader.readNext()) != null) {
                records.add(Arrays.asList(values));
            }
            return records;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (CsvValidationException e) {
            throw new RuntimeException(e);
        }
    }
    
    public static <T, U> List<U> convertStringListTodoubleList(List<T> listOfString,
                                                               Function<T, U> function) {
        return listOfString.stream()
            .map(function)
            .collect(Collectors.toList());
    }
    
    public static int countRowCsv(String path) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(path));
        String input;
        int count = 0;
        while ((input = bufferedReader.readLine()) != null) {
            count++;
        }
        return count;
    }
    
    public static List<DataSet> getDataSet(String csvPath, String fileDelimiter, int numLinesToSkip,
                                           int batchSize, int labelIndexFrom, int labelIndexTo) throws IOException, InterruptedException {
        
        RecordReader rr = new CSVRecordReader(numLinesToSkip, fileDelimiter);
        int rows = Util.countRowCsv(csvPath);
        rr.initialize(new FileSplit(new File(csvPath)));

        DataSetIterator iter =
            new RecordReaderDataSetIterator.Builder(rr, batchSize).regression(labelIndexFrom,labelIndexTo).build();
        
        List<DataSet> dataset = new ArrayList<>();
        
        while (iter.hasNext()) {
            DataSet batch = iter.next();
            dataset.add(batch);
        }
        Collections.shuffle(dataset);
        return dataset;
    }
    
    public static void writeArray(List<Double> avgError, List<Double> avgAccuracy, String fileName, String modelStructure)
        throws IOException {
        
        FileWriter fw = new FileWriter(fileName, true);
        BufferedWriter bw = new BufferedWriter(fw);
        
        Double[] lr = {0.003, 0.01, 0.03, 0.1, 0.3};
        Double[] mr = {0.003, 0.01, 0.03, 0.1, 0.3};
        List<Double> lrs = new ArrayList<Double>(Arrays.asList(lr));
        
        bw.write(modelStructure + "\n");
        bw.write("AvgError: [\n");
        
        int length = lrs.size();
        
        for (int i = length - 1; i >= 0; --i) {
            bw.write("[");
            String comma_out = ",\n";
            if(i == 0) {
                comma_out = "\n";
            }
            
            for (int j = 0; j < length; ++j) {
                String comma = ",";
                if(j == length-1) {
                    comma = "";
                }
                int ind = i+j*length;
                bw.write(avgError.get(ind) + comma);
            }
            bw.write("]" + comma_out);
        }
        bw.write("]\n");
        
        if(avgAccuracy == null){
            bw.close();
            return;
        }
        bw.write("AvgAcc: [\n");
        
        for (int i = length - 1; i >= 0; --i) {
            bw.write("[");
            String comma_out = ",\n";
            if(i == 0) {
                comma_out = "\n";
            }
            for (int j = 0; j < length; ++j) {
                String comma = ",";
                if(j == length-1) {
                    comma = "";
                }
                int ind = i+j*length;
                bw.write(avgAccuracy.get(ind) + comma);
            }
            bw.write("]" + comma_out);
        }
        bw.write("]\n");
        bw.close();
    }
    
    
}
