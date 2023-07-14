package com.ci.activation_function;

public class ReLUFunction implements ActivationFunction {
    @Override
    public Double activate(Double b) {
        return Math.max(0.0, b);
    }
    
    @Override
    public Double activateDiff(Double b) {
        return b <= 0.0 ? 0.0 : 1.0;
    }
}
