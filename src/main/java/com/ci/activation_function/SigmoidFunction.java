package com.ci.activation_function;

public class SigmoidFunction implements ActivationFunction {
    @Override
    public Double activate(Double b) {
        return 1.0 / (1.0 + Math.exp(-b));
    }
    
    @Override
    public Double activateDiff(Double b) {
        return activate(b) * (1.0 - activate(b));
    }
    
    @Override
    public String toString() {
        return "Sigmoid Function";
    }
}
