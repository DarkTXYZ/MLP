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
    
    private ActivationFunction activationFunction;
    
    public static void main(String[] args) {

    }
}
