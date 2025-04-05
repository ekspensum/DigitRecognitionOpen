package pl.aticode;

import pl.aticode.data.Image;

import java.util.ArrayList;
import java.util.List;

public class Util {

    private static final double LAMBDA = 1.0507;
    private static final double ALPHA = 1.67326;

    public static List<Image> normalizeImage(List<Image> imageList, int rate) {
        for (Image image : imageList) {
            int imageWidth = image.getData().length;
            int imageHigh = image.getData()[0].length;
            double[][] imageNormalizedMatrix = new double[imageWidth][imageHigh];
            for (int i = 0; i < imageWidth; i++) {
                for (int j = 0; j < imageHigh; j++) {
                    imageNormalizedMatrix[i][j] = image.getData()[i][j] / rate;
                }
            }
            image.setData(imageNormalizedMatrix);
        }

        return imageList;
    }

    public static double[] matrixListToVector(List<double[][]> matrixList) {
        var matrixSize = matrixList.stream().findAny().orElseThrow();
        var vector = new double[matrixSize.length * matrixSize[0].length * matrixList.size()];
        int index = 0;
        for (double[][] matrix : matrixList) {
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix.length; j++) {
                    vector[index] = matrix[i][j];
                    index++;
                }
            }
        }
        return vector;
    }

    public static List<double[][]> vectorToMatrixList(double[] vector, int width, int high, int filterSize) {
        var matrixList = new ArrayList<double[][]>();
        for (int x = 0; x < filterSize; x++) {
            var matrix = new double[width][high];
            int index = 0;
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < high; j++) {
                    matrix[i][j] = vector[index];
                    index++;
                }
            }
            matrixList.add(matrix);
        }
        return matrixList;
    }

}
