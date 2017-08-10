package com.aviv871.edu.AI871;

import java.util.ArrayList;
import java.util.Arrays;

public class TrainingSet
{
    private final int INPUT_SIZE;
    private final int OUTPUT_SIZE;

    //double[][] <- index 0 for inputs, index 1 for outputs in the first part - the second part for values
    private ArrayList<double[][]> data = new ArrayList<>();

    public TrainingSet(int inputSize, int outputSize)
    {
        this.INPUT_SIZE = inputSize;
        this.OUTPUT_SIZE = outputSize;
    }

    public void addData(double[] input, double[] target)
    {
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return;
        data.add(new double[][]{input, target});
    }

    public TrainingSet extractBatch(int size) // Extract random data as a new TrainingSet (in the given size)
    {
        if(size > 0 && size <= this.getSize())
        {
            TrainingSet set = new TrainingSet(INPUT_SIZE, OUTPUT_SIZE);
            Integer[] ids = NetworkUtils.randomValues(0, this.getSize() - 1, size);
            for(Integer i: ids)
            {
                set.addData(this.getInput(i), this.getOutput(i));
            }
            return set;
        }
        else return this;
    }

    public String toString()
    {
        StringBuilder s = new StringBuilder("TrainingSet [" + INPUT_SIZE + " ; " + OUTPUT_SIZE + "]\n");

        int index = 0;
        for(double[][] r: data)
        {
            s.append(index).append(":   ").append(Arrays.toString(r[0])).append("  >-||-<  ").append(Arrays.toString(r[1])).append("\n");
            index++;
        }

        return s.toString();
    }

    public int getSize()
    {
        return data.size();
    }

    public double[] getInput(int index)
    {
        if(index >= 0 && index < getSize()) return data.get(index)[0];
        else return null;
    }

    public double[] getOutput(int index)
    {
        if(index >= 0 && index < getSize()) return data.get(index)[1];
        else return null;
    }

    public int getInputSize()
    {
        return INPUT_SIZE;
    }

    public int getOutputSize()
    {
        return OUTPUT_SIZE;
    }
}
