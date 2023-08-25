package com.ci;

import com.ci.activation_function.ActivationFunctionFactory;
import com.ci.loss_function.LossFunction;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

@Getter
@Setter
public class MLP {
    private List<HiddenLayer> layerList = new ArrayList<>();
    private INDArray input;
    private INDArray output;
    private INDArray outputDenormalized;
    private LossFunction lossFunction;
    private Double learningRate;
    private Double momentumRate;
    private int datasetSize;
    private Double averageError = 0.0;
    private Double mean = 0.0;
    private Double std = 1.0;


    public MLP(String input, LossFunction lossFunction, Double learningRate, Double momentumRate) {

        this.learningRate = learningRate;
        this.momentumRate = momentumRate;
        this.lossFunction = lossFunction;

        String[] parse = input.split("-");
        int prevLayerNode = 0;
        Integer index = 0;
        for (String layerInput : parse) {
            if (index == 0) {
                prevLayerNode = Integer.parseInt(parse[0].substring(0, 1));
                index++;
                continue;
            }
            int numberOfPerceptron =
                Integer.parseInt(layerInput.substring(0, layerInput.length() - 1));
            HiddenLayer layer = new HiddenLayer(index.toString(), prevLayerNode, numberOfPerceptron,
                ActivationFunctionFactory.generate(
                    layerInput.substring(layerInput.length() - 1)));
            this.layerList.add(layer);
            prevLayerNode = numberOfPerceptron;
            index++;

        }
    }

    public void train(List<DataSet> trainDatas) {
        int index = 0;

        this.averageError = 0.0;
        int trainIteration = 0;
        int dataInd = 0;
        int dataSize = trainDatas.size();
        double sumErrorOutput = 0.0;
        double sumError = 0.0;

        do {
            if (dataInd >= dataSize)
                dataInd = 0;

            DataSet trainData = trainDatas.get(dataInd);
            INDArray input = Nd4j.toFlattened(trainData.getFeatures());
            INDArray desiredOutput = Nd4j.toFlattened(trainData.getLabels());

            input = input.sub(mean).div(std);
            desiredOutput = desiredOutput.sub(mean).div(std);

            this.forward(input);
            this.backward(lossFunction.errorDiff(this.getOutput(), desiredOutput));

            sumErrorOutput += lossFunction.error(this.getOutput(), desiredOutput).sum().getDouble();
            sumError +=
                lossFunction.error(this.getOutputDenormalized(), desiredOutput.mul(std).add(mean))
                    .sum().getDouble();
            trainIteration++;
            dataInd++;

            this.averageError = sumErrorOutput / trainIteration;

        } while (this.averageError >= 0.00001 && trainIteration < 1000);

        System.out.println("Train Average Error: " + this.averageError);
        System.out.println("Train Average " + lossFunction + ": " + (sumError / trainIteration));
    }

    public void test(List<DataSet> testDatas) {
        Double sumError = 0.0;
        Double MAE = 0.0;
        for (DataSet testData : testDatas) {
            INDArray input = Nd4j.toFlattened(testData.getFeatures());
            INDArray desiredOutput = Nd4j.toFlattened(testData.getLabels());

            input = input.sub(mean).div(std);
            this.forward(input);

            sumError +=
                lossFunction.error(this.getOutputDenormalized(), desiredOutput).sum().getDouble();
        }
        System.out.println("Test Average " + lossFunction + ": " + (sumError / testDatas.size()));
    }

    public void forward(INDArray input) {
        this.input = input;

        INDArray outputLayer = input;
        for (HiddenLayer layer : layerList) {
            layer.forward(outputLayer);
            outputLayer = layer.getOutputs();
        }
        this.output = outputLayer;
        this.outputDenormalized = outputLayer.mul(std).add(mean);
    }

    public void backward(INDArray errors) {
        HiddenLayer prevLayer = null;

//      Calculate Local Gradients
        for (int i = layerList.size() - 1; i >= 0; --i) {
            HiddenLayer layer = layerList.get(i);
            if (i == layerList.size() - 1) {
                layer.backwardOutputLayer(errors);
            } else {
                List<Perceptron> perceptrons = layer.getPerceptronList();
                for (int j = 0; j < perceptrons.size(); ++j) {
                    layer.backward(j, prevLayer.getLocalGradients(),
                        prevLayer.getWeightsFromPerceptron(j));
                }
            }

            layer.generateLocalGradients();
            prevLayer = layer;
        }

//      Update Weights
        for (int i = layerList.size() - 1; i >= 0; --i) {
            HiddenLayer layer = layerList.get(i);
            if (i == 0) {
                layer.updateWeight(this.input, this.learningRate, this.momentumRate);
            } else {
                HiddenLayer layerBefore = layerList.get(i - 1);
                layer.updateWeight(layerBefore.getOutputs(), this.learningRate, this.momentumRate);
            }
            layer.updateBias(this.learningRate);
        }
    }

    @Override
    public String toString() {
        return "MLP{" +
            "hiddenLayerList=" + layerList + "\n" +
            "avg error= " + this.averageError + "\n" +
            '}';
    }
}
