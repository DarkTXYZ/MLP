package com.ci.activation_function;

public class TanhFunction implements ActivationFunction {
    @Override
    public Double activate(Double b) {
        return (Math.exp(b) - Math.exp(-b)) / (Math.exp(b) + Math.exp(-b));
    }
    
    @Override
    public Double activateDiff(Double b) {
        return 1.0 - Math.pow(activate(b), 2);
    }
}
