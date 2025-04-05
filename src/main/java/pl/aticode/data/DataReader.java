package pl.aticode.data;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataReader {

    private final int rows = 28;
    private final int cols = 28;

    public List<Image> readData() throws FileNotFoundException {
        List<Image> imageList = new ArrayList<>();
        Scanner scanner = new Scanner(new File("DataSet/mnist_train.csv"));
        scanner.useDelimiter(",");
        while (scanner.hasNextLine()) {
            String[] nextLine = scanner.nextLine().split(",");
            int label = Integer.parseInt(nextLine[0]);
            double[][] data = new double[rows][cols];
            int lineIndex = 1;
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    data[i][j] = Double.parseDouble(nextLine[lineIndex]);
                    lineIndex++;
                }
            }
            imageList.add(new Image(data, label));
        }
        scanner.close();
        return imageList;
    }
}
