package com.aviv871.edu.AI871;

import com.aviv871.edu.Math871.Math871;

import java.util.Arrays;

public class BasicNetwork // Feedforward network with backpropagation
{
    // Stores the size of each layer (number of neurons)
    private final int[] NETWORK_LAYER_SIZES;
    // Stores the size the input layer (number of neurons)
    private final int INPUT_SIZE;
    // Stores the size the output layer (number of neurons)
    private final int OUTPUT_SIZE;
    // Stores the size the network (number of layers)
    private final int NETWORK_SIZE;

    // Stores the outputs of the neurons. the first dimension of the array is the layer and the second is the index of the neuron inside the layer
    private double[][] outputs;
    // Stores the weights of the lines between the neurons. the first dimension of the array is the layer of this neuron, the second is the index
    // of the neuron inside his layer and the third is the index (inside his layer) of the neuron in the other side of the line - in the last layer
    private double[][][] weights;
    // Stores the bias of the neurons. the first dimension of the array is the layer of the neuron and the second is the index of the neuron inside his layer
    private double[][] bias;

    // Something
    private double learningRatio = 0.1;

    public BasicNetwork(int[] networkLayerSizes)
    {
        this.NETWORK_LAYER_SIZES = networkLayerSizes;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[this.NETWORK_SIZE - 1];

        this.outputs = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        for(int i = 0; i < NETWORK_SIZE; i++)
        {
            this.outputs[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.bias[i] = NetworkUtils.createRandomArray(NETWORK_LAYER_SIZES[i], 0.3, 0.7);

            if(i > 0) // The first layer (input layer) has no weights\lines from previous layer
            {
                this.weights[i] = NetworkUtils.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i-1], -0.3, 0.5);
            }
        }
    }

    public double[] calculate(double[] input)
    {
        if(input.length != INPUT_SIZE) return null;

        this.outputs[0] = input;
        for(int layer = 1; layer < NETWORK_SIZE; layer++)
        {
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
            {
                double sum = bias[layer][neuron];
                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++)
                {
                    sum += outputs[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }

                outputs[layer][neuron] = Math871.sigmoid(sum);
            }
        }

        //System.out.println("Output: " + Arrays.toString(outputs[NETWORK_SIZE-1]));
        return outputs[NETWORK_SIZE-1]; // Return only the output layer
    }

    public void train(double[] input, double[] target)
    {
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return;

        calculate(input);
        adjustWeights(target);
    }

    public void train(TrainingSet trainingSet)
    {
        if(trainingSet.getInputSize() != INPUT_SIZE || trainingSet.getOutputSize() != OUTPUT_SIZE) return;

        for(int i = 0; i < trainingSet.getSize(); i++)
        {
            calculate(trainingSet.getInput(i));
            adjustWeights(trainingSet.getOutput(i));
        }
    }

    private void adjustWeights(double[] target) // By backpropagation
    {
        double[][] errors = new double[NETWORK_SIZE][];
        for(int i = 0; i < NETWORK_SIZE; i++)
        {
            errors[i] = new double[NETWORK_LAYER_SIZES[i]];
        }

        for (int layer = NETWORK_SIZE - 1; layer > 0; layer--) // Input layer has no calculation in it
        {
            if (layer == NETWORK_SIZE - 1) // Layer is output
            {
                for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
                {
                    errors[layer][neuron] = (outputs[layer][neuron] - target[neuron]) * Math871.sigmoidPrime(outputs[layer][neuron]);
                    double delta = learningRatio * errors[layer][neuron];

                    for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++)
                    {
                        weights[layer][neuron][prevNeuron] -= outputs[layer-1][prevNeuron] * delta;
                    }

                    bias[layer][neuron] -= delta;
                }
            }
            else // Layer is hidden layer
            {
                for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
                {
                    for(int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron++)
                    {
                        errors[layer][neuron] += weights[layer+1][nextNeuron][neuron] * errors[layer+1][nextNeuron];
                    }
                    errors[layer][neuron] *= Math871.sigmoidPrime(outputs[layer][neuron]);

                    double delta = learningRatio * errors[layer][neuron];
                    for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++)
                    {
                        weights[layer][neuron][prevNeuron] -= outputs[layer-1][prevNeuron] * delta;
                    }

                    bias[layer][neuron] -= delta;
                }
            }
        }
    }

    public void setLearningRatio(double value)
    {
        this.learningRatio = value;
    }
}