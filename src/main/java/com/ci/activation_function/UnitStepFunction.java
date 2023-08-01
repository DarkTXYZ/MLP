package com.ci.activation_function;

public class UnitStepFunction implements ActivationFunction{
    @Override
    public Double activate(Double b) {
        return b < 0 ? 0.0 : 1.0;
    }
    
    @Override
    public Double activateDiff(Double b) {
        return 0.0;
    }
    
    @Override
    public String toString() {
        return "Unit Step Function";
    }
}
