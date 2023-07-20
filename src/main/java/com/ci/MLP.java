package com.ci;

import com.ci.HiddenLayer;
import com.ci.activation_function.ActivationFunctionFactory;

import java.util.List;

public class MLP {
    private List<HiddenLayer> hiddenLayerList;
    

    public MLP(String input) {
        String[] parse = input.split("-");
        int prev = 0;
        for(String layerInput : parse) {
            int numberOfPerceptron = Integer.parseInt(layerInput.substring(0 , layerInput.length() - 1));
            HiddenLayer layer = new HiddenLayer(prev, numberOfPerceptron, ActivationFunctionFactory.generate(layerInput.substring(layerInput.length() - 1)));
        }
    }
    
    
}
