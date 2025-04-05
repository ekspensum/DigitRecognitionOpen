package pl.aticode;

import pl.aticode.data.DataReader;
import pl.aticode.data.Image;
import pl.aticode.network.NeuralNetwork;

import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        try {
            DataReader dataReader = new DataReader();
            List<Image> imageList = dataReader.readData();
            List<Image> normalizedImage = Util.normalizeImage(imageList, 25600);
            Collections.shuffle(normalizedImage);

            var neuralNetwork = new NeuralNetwork.NetworkBuilder()
                    .withImageList(normalizedImage)
                    .withParameters(10)
                    .withConvolutionalLayer(12, 5, 5, 1, 1, 0.05)
                    .withPoolingLayer(3, 2)
                    .withFullyConnectedLayer(0.1, 10)
                    .build();
            neuralNetwork.train();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
