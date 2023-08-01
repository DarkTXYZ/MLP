package com.ci;

import com.ci.activation_function.ActivationFunction;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Getter
@Setter
@Builder
public class Perceptron {
    private INDArray weights;
    private INDArray prevWeights;
    private Double bias;
    private Double value;
    private Double output;
    private Double localGradient;
    
    private ActivationFunction activationFunction;
    
    public Double calculateOutput(INDArray input) {
        INDArray dotProduct = input.mul(weights);
        this.value = dotProduct.sum().getDouble();
//        System.out.println(dotProduct);
        this.output = this.activationFunction.activate(this.value + this.bias);
        return this.output;
    }
    
    public void calculateLocalGradient(Double error) {


//        System.out.println(this.activationFunction.activateDiff(this.value+this.bias));
        this.localGradient = this.activationFunction.activateDiff(this.value + this.bias) * error;
//        System.out.println("----");
//        System.out.println(this.value);
//        System.out.println(this.bias);
//        System.out.println(error);
//        System.out.println(this.localGradient);
//        System.out.println("----");
//        System.out.println(this.localGradient);
        
    }
    
    @Override
    public String toString() {
        return "Perceptron{" + "\n" +
            "\tweights = " + weights + "\n" +
            "\tbias = " + bias + "\n" +
            "\toutput = " + output + "\n" +
            "\tactivationFunction = " + activationFunction + "\n" +
            '}' + "\n";
    }
    
    public void updateWeight(INDArray layerBefore, Double learningRate, Double momentumRate) {
        
        INDArray localG = Nd4j.zeros(weights.length()).addi(this.localGradient);

//        System.out.println(localG);
//        System.out.println(layerBefore);
//        System.out.println(localG.mul(layerBefore));
//        System.out.println(learningRate);
//        System.out.println(localG.mul(layerBefore).mul(learningRate));
        
        INDArray deltaWeight = this.weights.sub(this.prevWeights);
        INDArray updatedWeight = this.weights.sub(localG.mul(layerBefore).mul(learningRate)).sub(deltaWeight.mul(momentumRate));
        this.prevWeights = this.weights;
        this.weights = updatedWeight;
        
    }
    
    public void updateBias(Double learningRate) {
//        System.out.println("New Bias = " + this.bias + " - " + learningRate + " * " + this.localGradient);
        this.bias = this.bias - learningRate * this.localGradient;
    }
}
