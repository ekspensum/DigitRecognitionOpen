package pl.aticode.network;

import lombok.Getter;
import pl.aticode.data.Image;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer {

    @Getter
    private final List<double[][]> filterList;
    private final int filterWidth;
    private final int filterHigh;
    private final int padding;
    private final int stride;
    @Getter
    private final int imageWidth;
    @Getter
    private final int imageHigh;
    private final double learnFactor;
    @Getter
    private final int outputWidth;
    @Getter
    private final int outputHeight;
    private final int dOutputWidth;
    private final int dOutputHeight;
    private Image currentImage;
    private final Random random = new Random();


    public ConvolutionLayer(int filtersNumber, int filterWidth, int filterHigh, int padding, int stride, int imageWidth, int imageHigh, double learnFactor) {
        this.filterWidth = filterWidth;
        this.filterHigh = filterHigh;
        this.padding = padding;
        this.stride = stride;
        this.imageWidth = imageWidth;
        this.imageHigh = imageHigh;
        this.learnFactor = learnFactor;
        this.outputWidth = (this.imageWidth + this.padding * 2 - filterWidth) / stride + 1;
        this.outputHeight = (this.imageHigh + this.padding * 2 - filterHigh) / stride + 1;
        this.dOutputWidth = (this.outputWidth - 1) * this.stride + 1;
        this.dOutputHeight = (this.outputHeight - 1) * this.stride + 1;
        this.filterList = createFilterList(filtersNumber);
    }

    public List<double[][]> convolution(Image image) {
        this.currentImage = image;
        List<double[][]> activationMapList = new ArrayList<>();
        for (double[][] filter : this.filterList) {
            var activationMap = convolve(image.getData(), filter);
            activationMapList.add(activationMap);
        }
        return activationMapList;
    }

    public double[][] convolve(double[][] imageMatrix, double[][] filter) {
        var activationMap = new double[this.outputWidth][this.outputHeight];
        for (int x = 0; x < imageMatrix.length - this.filterWidth; x++) {
            for (int y = 0; y < imageMatrix[0].length - this.filterHigh; y++) {
                double convolutionSum = 0.0;
                for (int i = 0; i < this.filterWidth; i++) {
                    for (int j = 0; j < this.filterHigh; j++) {
                        convolutionSum += imageMatrix[i + x][j + y] * filter[i][j];
                    }
                }
                activationMap[x][y] = convolutionSum;
            }
        }
        return activationMap;
    }

    public void backPropagation(List<double[][]> dPoolingLayerOutput) {
        for (int i = 0; i < this.filterList.size(); i++) {
            double[][] error = dPoolingLayerOutput.get(i);
            double[][] errorMoved = moveWithStride(error);
            double[][] dCurrentImage = convolve(this.currentImage.getData(), errorMoved);
            double[][] delta = multiply(dCurrentImage, this.learnFactor);
            double[][] filterModified = add(this.filterList.get(i), delta);
            this.filterList.set(i, filterModified);
        }
    }

    private List<double[][]> createFilterList(int filtersNumber) {
        List<double[][]> filters = new ArrayList<>();
        for (int f = 0; f < filtersNumber; f++) {
            double[][] filter = new double[this.filterWidth][this.filterHigh];
            for (int i = 0; i < this.filterWidth; i++) {
                for (int j = 0; j < this.filterHigh; j++) {
                    filter[i][j] = this.random.nextGaussian();
                }
            }
            filters.add(filter);
        }
        return filters;
    }

    public double[][] addPaddingToImage(double[][] image) {
        var output = new double[this.imageWidth + this.padding * 2][this.imageHigh + this.padding * 2];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                if (i < this.padding || j < this.padding || i > this.imageWidth || j > this.imageHigh) {
                    output[i][j] = 0.0;
                } else {
                    output[i][j] = image[i - this.padding][j - this.padding];
                }
            }
        }
        return output;
    }

    private double[][] moveWithStride(double[][] dInput) {
        if (this.stride == 1) {
            return dInput;
        }
        double[][] dOutput = new double[this.dOutputWidth][this.dOutputHeight];
        for (int i = 0; i < dInput.length; i++) {
            for (int j = 0; j < dInput[0].length; j++) {
                dOutput[i * this.stride][j * this.stride] = dInput[i][j];
            }
        }
        return dOutput;
    }

    private double[][] add(double[][] matrix1, double[][] matrix2) {
        double[][] output = new double[matrix1.length][matrix1[0].length];
        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix1[0].length; j++) {
                output[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        return output;
    }

    private double[][] multiply(double[][] imageData, double scalar) {
        double[][] output = new double[imageData.length][imageData[0].length];
        for (int i = 0; i < imageData.length; i++) {
            for (int j = 0; j < imageData[0].length; j++) {
                output[i][j] = imageData[i][j] * scalar;
            }
        }
        return output;
    }
}
