package com.ci.activation_function;

public class ReLUFunction implements ActivationFunction {
    @Override
    public Double activate(Double b) {
        return b >= 0 ? b : 0.01 * b;
    }
    
    @Override
    public Double activateDiff(Double b) {
        return b <= 0.0 ? 0.01 : 1.0;
    }
    
    @Override
    public String toString() {
        return "ReLU Function";
    }
}
