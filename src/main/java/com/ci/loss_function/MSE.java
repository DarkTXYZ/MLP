package com.ci.loss_function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MSE implements LossFunction{
    @Override
    public INDArray error(INDArray pred, INDArray desiredOutput) {
        return Transforms.pow(desiredOutput.sub(pred), 2).div(desiredOutput.length());
        
    }
    
    @Override
    public INDArray errorDiff(INDArray pred, INDArray desiredOutput) {
        return desiredOutput.sub(pred).mul(-2);
    }
    
    @Override
    public String toString() {
        return "MSE";
    }
}
