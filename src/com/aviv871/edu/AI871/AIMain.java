package com.aviv871.edu.AI871;

import java.util.Arrays;

public class AIMain
{
    public static void main(String[] args)
    {
        // Creating the network
        int[] layers = {4, 3, 1};
        BasicNetwork network = new BasicNetwork(layers);
        network.setLearningRatio(0.1);

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
