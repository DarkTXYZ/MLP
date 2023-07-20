package com.ci.activation_function;

public class ActivationFunctionFactory {
    
    //      I - Identity
    //      R - ReLU
    //      S - Sigmoid
    //      T - Tanh
    //      U - Unit Step
    
    public static ActivationFunction generate(String type) {
        switch (type) {
            case "I" -> {
                return new IdentityFunction();
            }
            case "R" -> {
                return new ReLUFunction();
            }
            case "S" -> {
                return new SigmoidFunction();
            }
            case "T" -> {
                return new TanhFunction();
            }
            case "U" -> {
                return new UnitStepFunction();
            }
        }
        return new IdentityFunction();
    }
    
}