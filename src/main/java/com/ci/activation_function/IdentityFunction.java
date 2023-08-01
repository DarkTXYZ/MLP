package com.ci.activation_function;

public class IdentityFunction implements ActivationFunction {
    @Override
    public Double activate(Double b) {
        return b;
    }
    
    @Override
    public Double activateDiff(Double b) {
        return 1.0;
    }
    
    @Override
    public String toString() {
        return "Identity Function";
    }
}
