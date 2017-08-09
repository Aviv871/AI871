package com.aviv871.edu.AI871;

public class NetworkUtils
{
    public static double[] createArray(int size, double init_value)
    {
        if(size < 1)
        {
            return null;
        }

        double[] arr = new double[size];
        for(int i = 0; i < size; i++)
        {
            arr[i] = init_value;
        }

        return arr;
    }

    public static double[] createRandomArray(int size, double lower_bound, double upper_bound)
    {
        if(size < 1)
        {
            return null;
        }

        double[] arr = new double[size];
        for(int i = 0; i < size; i++)
        {
            arr[i] = randomValue(lower_bound, upper_bound);
        }

        return arr;
    }

    public static double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound)
    {
        if(sizeX < 1 || sizeY < 1)
        {
            return null;
        }

        double[][] arr = new double[sizeX][sizeY];
        for(int i = 0; i < sizeX; i++)
        {
            arr[i] = createRandomArray(sizeY, lower_bound, upper_bound);
        }

        return arr;
    }

    public static double randomValue(double lower_bound, double upper_bound)
    {
        return Math.random() * (upper_bound - lower_bound) + lower_bound;
    }

    public static Integer[] randomValues(int lowerBound, int upperBound, int amount) // Array of integers without the same value twice
    {
        lowerBound--;

        if(amount > (upperBound - lowerBound))
        {
            return null;
        }

        Integer[] values = new Integer[amount];
        for(int i = 0; i < amount; i++)
        {
            int n = (int) (Math.random() * (upperBound - lowerBound + 1) + lowerBound);
            while(containsValue(values, n))
            {
                n = (int) (Math.random() * (upperBound - lowerBound + 1) + lowerBound);
            }

            values[i] = n;
        }

        return values;
    }

    public static <T extends Comparable<T>> boolean containsValue(T[] arr, T value)
    {
        for(int i = 0; i < arr.length; i++)
        {
            if(arr[i] != null)
            {
                if(value.compareTo(arr[i]) == 0)
                {
                    return true;
                }
            }
        }

        return false;
    }

    public static int indexOfHighestValue(double[] values)
    {
        int index = 0;
        for(int i = 1; i < values.length; i++)
        {
            if(values[i] > values[index])
            {
                index = i;
            }
        }

        return index;
    }
}
