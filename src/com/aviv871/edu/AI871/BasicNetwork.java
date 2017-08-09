package com.aviv871.edu.AI871;

import com.aviv871.edu.Math871.Math871;

import java.util.Arrays;

public class BasicNetwork // Feedforward network with backpropagation
{
    //
    private static final double LEARNING_RATIO = 0.1;

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

    public void train(double[] input, double[] goodOutput)
    {
        if(input.length != INPUT_SIZE) return;
        if(goodOutput.length != OUTPUT_SIZE) return;

        calculate(input);
        adjustWeights(goodOutput);
    }

    private void adjustWeights(double[] goodOutput)
    {
        double[][] errors = new double[NETWORK_SIZE][];
        for(int i = 0; i < NETWORK_SIZE; i++)
        {
            errors[i] = new double[NETWORK_LAYER_SIZES[i]];
        }

        for (int layer = NETWORK_SIZE - 1; layer >= 0; layer--)
        {
            if(layer == 0) continue; // Input layer has no calculation in it

            if (layer == NETWORK_SIZE - 1) // Layer is output
            {
                for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
                {
                    errors[layer][neuron] = outputs[layer][neuron] - goodOutput[neuron];

                    for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++)
                    {
                        weights[layer][neuron][prevNeuron] -= LEARNING_RATIO * outputs[layer-1][prevNeuron] * errors[layer][neuron] * Math871.sigmoidPrime(outputs[layer][neuron]);
                        bias[layer][neuron] -= LEARNING_RATIO * errors[layer][neuron] * Math871.sigmoidPrime(outputs[layer][neuron]);
                    }
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

                    for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++)
                    {
                        weights[layer][neuron][prevNeuron] -= LEARNING_RATIO * outputs[layer-1][prevNeuron] * errors[layer][neuron] * Math871.sigmoidPrime(outputs[layer][neuron]);
                        bias[layer][neuron] -= LEARNING_RATIO * errors[layer][neuron] * Math871.sigmoidPrime(outputs[layer][neuron]);
                    }
                }
            }
        }
    }

    public static void main(String[] args)
    {
        // Creating the network
        int[] layers = {4, 3, 1};
        BasicNetwork network = new BasicNetwork(layers);

        // Training the network
        for(int i=0; i < 8000; i++)
        {
            double[] input1 = {1, 1, 1, 1};
            double[] output1 = {1};
            network.train(input1, output1);

            double[] input2 = {1, 1, 1, 0};
            double[] output2 = {1};
            network.train(input2, output2);

            double[] input3 = {1, 1, 0, 1};
            double[] output3 = {1};
            network.train(input3, output3);

            double[] input4 = {1, 1, 0, 0};
            double[] output4 = {0};
            network.train(input4, output4);

            double[] input5 = {1, 0, 1, 1};
            double[] output5 = {1};
            network.train(input5, output5);

            double[] input6 = {1, 0, 1, 0};
            double[] output6 = {0};
            network.train(input6, output6);

            double[] input7 = {1, 0, 0, 1};
            double[] output7 = {0};
            network.train(input7, output7);

            double[] input8 = {1, 0, 0, 0};
            double[] output8 = {0};
            network.train(input8, output8);

            double[] input9 = {0, 1, 1, 1};
            double[] output9 = {1};
            network.train(input9, output9);

            double[] input10 = {0, 1, 1, 0};
            double[] output10 = {0};
            network.train(input10, output10);

            double[] input11 = {0, 1, 0, 1};
            double[] output11 = {0};
            network.train(input11, output11);

            double[] input12 = {0, 1, 0, 0};
            double[] output12 = {0};
            network.train(input12, output12);

            double[] input13 = {0, 0, 1, 1};
            double[] output13 = {0};
            network.train(input13, output13);

            double[] input14 = {0, 0, 1, 0};
            double[] output14 = {0};
            network.train(input14, output14);

            double[] input15 = {0, 0, 0, 1};
            double[] output15 = {0};
            network.train(input15, output15);

            double[] input16 = {0, 0, 0, 0};
            double[] output16 = {0};
            network.train(input16, output16);
        }

        // Testing the network
        double[] input = {1,0,1,1};
        double[] inputt = {0,0,1,0};
        double[] inputtt = {1,1,1,1};
        double[] inputttt = {1,0,0,1};

        System.out.println("Test 1: " + Arrays.toString(network.calculate(input)));
        System.out.println("Test 2: " + Arrays.toString(network.calculate(inputt)));
        System.out.println("Test 3: " + Arrays.toString(network.calculate(inputtt)));
        System.out.println("Test 4: " + Arrays.toString(network.calculate(inputttt)));
    }
}