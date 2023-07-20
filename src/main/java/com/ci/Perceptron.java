package com.ci;

import com.ci.activation_function.ActivationFunction;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

@Getter
@Setter
@Builder
public class Perceptron {
    private INDArray weights;
    private Double bias;
    private Double output;
    
    private ActivationFunction activationFunction;
    
    public Double calculateOutput(INDArray input) {
        INDArray dotProduct = input.mul(weights);
        Double sumDotProduct = dotProduct.sum().getDouble();
        this.output = this.activationFunction.activate(sumDotProduct + bias);
        return this.output;
    }
}
