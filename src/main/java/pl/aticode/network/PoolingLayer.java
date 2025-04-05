package pl.aticode.network;

import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

public class PoolingLayer {

    @Getter
    private final int poolMatrixSize;
    private final int stride;
    private int[][] maxIndicesMap;
    @Getter
    private final int outputWidth;
    @Getter
    private final int outputHeight;
    private final int dInputHeight;
    private final int dInputWidth;

    public PoolingLayer(int poolMatrixSize, int stride, int convolvedImageWidth, int convolvedImageHigh) {
        this.poolMatrixSize = poolMatrixSize;
        this.stride = stride;
        this.outputWidth = (convolvedImageWidth - this.poolMatrixSize) / this.stride + 1;
        this.outputHeight = (convolvedImageHigh - this.poolMatrixSize) / this.stride + 1;
        this.dInputWidth = (this.outputWidth - 1) * stride + this.poolMatrixSize;
        this.dInputHeight = (this.outputHeight - 1) * stride + this.poolMatrixSize;
    }

    public List<double[][]> maxPooling(List<double[][]> activationMapList) {
        var maxPoolMatrixList = new ArrayList<double[][]>();
        for (double[][] convolvedImage : activationMapList) {
            var maxPoolMatrix = new double[this.outputWidth][this.outputHeight];
            this.maxIndicesMap = new int[this.outputWidth * this.outputWidth][2];
            for (int i = 0; i < this.outputWidth; i++) {
                for (int j = 0; j < this.outputHeight; j++) {
                    double maxValue = Double.NEGATIVE_INFINITY;
                    int maxRow = -1, maxCol = -1;
                    for (int x = 0; x < this.poolMatrixSize; x++) {
                        for (int y = 0; y < this.poolMatrixSize; y++) {
                            int row = i * this.stride + x;
                            int col = j * this.stride + y;
                            if (convolvedImage[row][col] > maxValue) {
                                maxValue = convolvedImage[row][col];
                                maxRow = row;
                                maxCol = col;
                            }
                        }
                    }
                    maxPoolMatrix[i][j] = maxValue;
                    this.maxIndicesMap[i * this.outputWidth + j] = new int[]{maxRow, maxCol};
                }
            }
            maxPoolMatrixList.add(maxPoolMatrix);
        }
        return maxPoolMatrixList;
    }

    public List<double[][]> backPropagation(List<double[][]> dFullyConnectedOutList) {
        var dPoolingOut = new ArrayList<double[][]>();
        for (double[][] dFullyConnectedOut : dFullyConnectedOutList) {
            double[][] dInput = new double[this.dInputHeight][this.dInputWidth];
            for (int i = 0; i < dFullyConnectedOut.length; i++) {
                for (int j = 0; j < dFullyConnectedOut[0].length; j++) {
                    int[] maxIndex = this.maxIndicesMap[i * dFullyConnectedOut[0].length + j];
                    dInput[maxIndex[0]][maxIndex[1]] += dFullyConnectedOut[i][j];
                }
            }
            dPoolingOut.add(dInput);
        }
        return dPoolingOut;
    }
}
