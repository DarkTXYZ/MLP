package com.ci.loss_function;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface LossFunction {
    INDArray error(INDArray pred , INDArray desiredOutput);
    
    INDArray errorDiff(INDArray pred , INDArray desiredOutput);
}
