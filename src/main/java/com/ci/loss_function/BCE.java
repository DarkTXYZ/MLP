package com.ci.loss_function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class BCE implements LossFunction {
    @Override
    public INDArray error(INDArray pred, INDArray desiredOutput) {
        return desiredOutput.mul(Transforms.log(pred))
            .add(desiredOutput.mul(-1.0).add(1.0).mul(Transforms.log(pred.mul(-1).add(1))))
            .mul(-1.0   );
    }
    
    @Override
    public INDArray errorDiff(INDArray pred, INDArray desiredOutput) {
        return pred.sub(desiredOutput).div(pred.mul(pred.mul(-1).add(1)));
    }
    
    @Override
    public String toString() {
        return "BCE";
    }
}
