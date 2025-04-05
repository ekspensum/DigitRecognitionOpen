package pl.aticode.data;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@AllArgsConstructor
public class Image {

    @Setter private double[][] data;
    private int label;
}
