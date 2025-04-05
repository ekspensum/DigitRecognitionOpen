package pl.aticode.network;

import pl.aticode.Util;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer {
    private final double learnFactor;
    private final int layerInputLength;
    private final int layerOutputLength;
    private final Random random;
    private double[] currentInput;
    private final double[][] weightsMatrix;
    private final double[] bias;

    public FullyConnectedLayer(double learnFactor, int layerInputLength, int layerOutputLength) {
        this.learnFactor = learnFactor;
        this.layerInputLength = layerInputLength;
        this.layerOutputLength = layerOutputLength;
        this.weightsMatrix = new double[this.layerOutputLength][this.layerInputLength];
        this.bias = new double[this.layerOutputLength];
        this.random = new Random();
        initialization();
    }

    public double[] feedForward(List<double[][]> maxPooling) {
        double[] maxPoolVector = Util.matrixListToVector(maxPooling);
        return forward(maxPoolVector);
    }

    public double[] forward(double[] maxPoolVector) {
        this.currentInput = maxPoolVector;
        double[] output = new double[this.layerOutputLength];
        for (int i = 0; i < this.layerOutputLength; i++) {
            output[i] = bias[i];
            for (int j = 0; j < this.layerInputLength; j++) {
                output[i] += this.weightsMatrix[i][j] * maxPoolVector[j];
            }
        }
        return output;
    }

    public double[] backPropagation(double[] unitErrorVector) {
        double[] dOutput = new double[this.layerInputLength];
        for (int i = 0; i < this.layerOutputLength; i++) {
            for (int j = 0; j < this.layerInputLength; j++) {
                dOutput[j] += this.weightsMatrix[i][j] * unitErrorVector[i];
                this.weightsMatrix[i][j] -= this.learnFactor * unitErrorVector[i] * this.currentInput[j];
            }
            this.bias[i] -= this.learnFactor * unitErrorVector[i];
        }
        return dOutput;
    }

    private void initialization() {
        for (int i = 0; i < this.layerOutputLength; i++) {
            for (int j = 0; j < this.layerInputLength; j++) {
                this.weightsMatrix[i][j] = this.random.nextDouble(-0.5, 0.5);
            }
            this.bias[i] = this.random.nextDouble(-0.5, 0.5);
        }
    }

}
