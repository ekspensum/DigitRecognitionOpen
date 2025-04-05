package pl.aticode.network;

import pl.aticode.Util;
import pl.aticode.data.Image;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private int epochs;
    private List<Image> imageList;
    private List<Image> imageWithPaddingList;
    private ConvolutionLayer convolutionLayer;
    private PoolingLayer poolingLayer;
    private FullyConnectedLayer fullyConnectedLayer;

    private NeuralNetwork() {
    }

    public void train() {
        for (int epoch = 0; epoch <= this.epochs; epoch++) {
            for (Image image : this.imageWithPaddingList) {
                double[] predicted = forwardPass(image);
                backPropagation(predicted, image.getLabel());
            }
            displayLoss(epoch);
        }
    }

    private double[] forwardPass(Image image) {
        List<double[][]> activationMapList = this.convolutionLayer.convolution(image);
        List<double[][]> maxPoolList = this.poolingLayer.maxPooling(activationMapList);
        return this.fullyConnectedLayer.feedForward(maxPoolList);
    }

    private void backPropagation(double[] predicted, int imageLabel) {
        double[] errors = calculateErrors(predicted, imageLabel);
        double[] fullyConnectedOutput = this.fullyConnectedLayer.backPropagation(errors);
        List<double[][]> fullyConnectedMatrixList = Util.vectorToMatrixList(fullyConnectedOutput, this.poolingLayer.getOutputWidth(), this.poolingLayer.getOutputHeight(),
                this.convolutionLayer.getFilterList().size());
        List<double[][]> poolingLayerOutput = this.poolingLayer.backPropagation(fullyConnectedMatrixList);
        this.convolutionLayer.backPropagation(poolingLayerOutput);
    }

    private double[] calculateErrors(double[] predicted, int imageLabel) {
        var actual = new double[predicted.length];
        actual[imageLabel] = -1;
        var unitErrorVector = new double[predicted.length];
        for (int i = 0; i < unitErrorVector.length; i++) {
            unitErrorVector[i] = actual[i] + predicted[i];
        }
        return unitErrorVector;
    }

    public void displayLoss(int epoch) {
        double correctPredicted = 0;
        for (Image image : this.imageWithPaddingList) {
            double[] predicted = forwardPass(image);
            double max = 0.0;
            int indexOfMax = 0;
            for (int i = 0; i < predicted.length; i++) {
                if (predicted[i] > max) {
                    max = predicted[i];
                    indexOfMax = i;
                }
            }
            if (indexOfMax == image.getLabel()) {
                correctPredicted++;
            }
        }
        double loss = (this.imageWithPaddingList.size() - correctPredicted) / this.imageWithPaddingList.size();
        System.out.printf("Epoch: %s | Correct predicted: %s | Loss: %.10f%n", epoch, correctPredicted, loss);
    }

    public static class NetworkBuilder {
        private final NeuralNetwork neuralNetwork;

        public NetworkBuilder() {
            this.neuralNetwork = new NeuralNetwork();
        }

        public NetworkBuilder withImageList(List<Image> normalizedImage) {
            this.neuralNetwork.imageList = normalizedImage;
            return this;
        }

        public NetworkBuilder withConvolutionalLayer(int filtersNumber, int filterWidth, int filterHigh, int padding, int stride, double learnFactor) {
            double[][] imageMatrix = this.neuralNetwork.imageList.stream().findFirst().orElseThrow().getData();
            this.neuralNetwork.convolutionLayer = new ConvolutionLayer(filtersNumber, filterWidth, filterHigh, padding, stride, imageMatrix[0].length, imageMatrix.length, learnFactor);
            if (padding > 0) {
                this.neuralNetwork.imageWithPaddingList = new ArrayList<>();
                for (Image image : this.neuralNetwork.imageList) {
                    double[][] imageWithPaddingMatrix = this.neuralNetwork.convolutionLayer.addPaddingToImage(image.getData());
                    this.neuralNetwork.imageWithPaddingList.add(new Image(imageWithPaddingMatrix, image.getLabel()));
                }
            } else {
                this.neuralNetwork.imageWithPaddingList = this.neuralNetwork.imageList;
            }
            return this;
        }

        public NetworkBuilder withPoolingLayer(int poolMatrixSize, int stride) {
            this.neuralNetwork.poolingLayer = new PoolingLayer(poolMatrixSize, stride, this.neuralNetwork.convolutionLayer.getOutputWidth(), this.neuralNetwork.convolutionLayer.getOutputHeight());
            return this;
        }

        public NetworkBuilder withFullyConnectedLayer(double learnFactor, int layerOutputLength) {
            int filterListSize = this.neuralNetwork.convolutionLayer.getFilterList().size();
            int layerInputLength = this.neuralNetwork.poolingLayer.getOutputWidth() * this.neuralNetwork.poolingLayer.getOutputHeight() * filterListSize;
            this.neuralNetwork.fullyConnectedLayer = new FullyConnectedLayer(learnFactor, layerInputLength, layerOutputLength);
            return this;
        }

        public NetworkBuilder withParameters(int epochs) {
            this.neuralNetwork.epochs = epochs;
            return this;
        }

        public NeuralNetwork build() {
            return this.neuralNetwork;
        }
    }
}
