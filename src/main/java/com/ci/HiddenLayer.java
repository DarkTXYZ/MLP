package com.ci;

import com.ci.activation_function.ActivationFunction;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class HiddenLayer {
    
    // generate array of output
    private List<Perceptron> perceptronList = new ArrayList<>();
    private INDArray outputs;
    private INDArray localGradients;
    private String name;
    private Integer prevLayer;
    
    public HiddenLayer(String name, int prevLayer, int n, ActivationFunction activationFunction) {
        this.name = name;
        
        for (int i = 0; i < n; ++i) {
//            INDArray randomWeight = Nd4j.rand(prevLayer);
            INDArray randomWeight = Nd4j.rand(prevLayer);
            INDArray zeros = Nd4j.zeros(prevLayer);
            
            Perceptron perceptron = Perceptron.builder()
                .activationFunction(activationFunction)
                .bias(0.0)
                .weights(randomWeight)
                .prevWeights(zeros)
                .build();
            perceptronList.add(perceptron);
        }
    }
    
    public void forward(INDArray outputsPrevLayer) {
        List<Double> temp = new ArrayList<>();
        for (Perceptron p : perceptronList) {
            temp.add(p.calculateOutput(outputsPrevLayer));
        }
        double[] arr = temp.stream().mapToDouble(Double::doubleValue).toArray();
        
        this.outputs = Nd4j.create(arr);
    }
    
    public void backward(int perceptronIndex, INDArray prevLocalGradients, INDArray prevWeights) {
        Perceptron target = perceptronList.get(perceptronIndex);
        
        INDArray dotProduct = prevLocalGradients.mul(prevWeights);
        Double sum = dotProduct.sum().getDouble();
//        System.out.println("Sum = " + sum);
        target.calculateLocalGradient(sum);
        
    }
    
    public void backwardOutputLayer(INDArray errors) {
        int index = 0;
        for (Perceptron p : perceptronList) {
            p.calculateLocalGradient(errors.getDouble(index));
            index++;
        }
    }
    
    public void generateLocalGradients() {
        List<Double> local = new ArrayList<>();
        for (Perceptron p : perceptronList) {
            local.add(p.getLocalGradient());
        }
        this.localGradients = Util.listToINDArray(local);
//        System.out.println(this.localGradients);
    }
    
    public INDArray getWeightsFromPerceptron(int index) {
        List<Double> weight = new ArrayList<>();
        for (Perceptron p : perceptronList) {
            weight.add(p.getWeights().getDouble(index));
        }
        return Util.listToINDArray(weight);
    }
    
    @Override
    public String toString() {
        return name + " Hidden Layer {\n" +
            perceptronList +
            ", outputs= " + outputs + "\n" +
            ", local gradients= " + localGradients + "\n" +
            '}';
    }
    
    public void updateWeight(INDArray layerBefore, Double learningRate, Double momentumRate) {
        for (Perceptron p : perceptronList) {
            p.updateWeight(layerBefore, learningRate, momentumRate);
        }
    }
    
    public void updateBias(Double learningRate) {
        for (Perceptron p : perceptronList) {
            p.updateBias(learningRate);
        }
    }
}
