package com.ci.loss_function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MAE implements LossFunction {
    @Override
    public INDArray error(INDArray pred, INDArray desiredOutput) {
        return Transforms.abs(desiredOutput.sub(pred)).div(pred.length());
    }
    
    @Override
    public INDArray errorDiff(INDArray pred, INDArray desiredOutput) {
        return pred.sub(desiredOutput).sum().getDouble() > 0.0 ? Nd4j.ones(1) : Nd4j.ones(1).mul(-1.0);
    }
    
    @Override
    public String toString() {
        return "MAE";
    }
}
