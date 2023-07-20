package com.ci;

import com.ci.activation_function.IdentityFunction;
import com.ci.activation_function.TanhFunction;
import com.ci.activation_function.UnitStepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        
//        Perceptron a = Perceptron.builder()
//            .activationFunction(new UnitStepFunction())
//            .build();
//
//        System.out.println(a.getActivationFunction().activate(-15.0));
        
//        INDArray a = Nd4j.linspace(2,6,10);
//        INDArray b = Nd4j.linspace(1,100,10);
//        System.out.println(a);
//        System.out.println(b);
//        System.out.println(a.mul(b));
//        System.out.println(a.mul(b).sum());

        INDArray ones = Nd4j.ones(2);

        HiddenLayer t = new HiddenLayer(2 , 5, new IdentityFunction());
        System.out.println(t.getPerceptronList());
        for(Perceptron p : t.getPerceptronList()) {
            System.out.println(p.getActivationFunction());
            System.out.println(p.getWeights());
            System.out.println(ones);
            System.out.println(p.calculateOutput(ones));
        }
        
        
//        List<Double> test = new ArrayList<>();
//        test.add(1.0);
//        test.add(2.0);
//
//        double[] arr = test.stream().mapToDouble(Double::doubleValue).toArray();
//
//        INDArray a = Nd4j.create(arr);
//        t.forward(a);
//        System.out.println(t.getOutputs());
        
    }
}