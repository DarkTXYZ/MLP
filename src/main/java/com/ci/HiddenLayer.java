package com.ci;

import com.ci.activation_function.ActivationFunction;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.NDArrayUtil;
import org.nd4j.list.NDArrayList;
import org.nd4j.nativeblas.Nd4jCpu;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class HiddenLayer {
    
    // generate array of output
    private List<Perceptron> perceptronList = new ArrayList<>();
    private INDArray outputs;
    
    public HiddenLayer(int prevLayer, int n, ActivationFunction activationFunction) {
    
        for (int i = 0; i < n; ++i) {
            INDArray randomWeight = Nd4j.rand(prevLayer);
        
            Perceptron perceptron = Perceptron.builder()
                .activationFunction(activationFunction)
                .bias(1.0)
                .weights(randomWeight)
                .build();
            perceptronList.add(perceptron);
        }
    }
    
    public void forward(INDArray outputsPrevLayer) {
        List<Double> temp = new ArrayList<>();
        for(Perceptron p : perceptronList) {
            temp.add(p.calculateOutput(outputsPrevLayer));
        }
        double[] arr = temp.stream().mapToDouble(Double::doubleValue).toArray();
        
        this.outputs = Nd4j.create(arr);
    }
}
