#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matavgKernel(int* my_array, int array_size)
{
    int i_index, j_index, temp;
    for (int i_index = 0; i_index < array_size; i_index++)
    {
        for (j_index = 0; j_index < array_size - 1; j_index++)
        {
            if (my_array[j_index] > my_array[j_index + 1])
            {
                temp = my_array[j_index];
                my_array[j_index] = my_array[j_index + 1];
                my_array[j_index + 1] = temp;
            }
        }
    }
    thrust::device_vector<int> d_vec(my_array, my_array + array_size);

    thrust::sort(d_vec.begin(), d_vec.end());
}

__global__ void matavgKernel2(int* array, int size)
{
    int thread_id = threadIdx.x;
	int temp;
	for (int i = 0; i < size - 1; i++)
	{
		if (array[thread_id] > array[thread_id + 1])
		{
			temp = array[thread_id];
			array[thread_id] = array[thread_id + 1];
			array[thread_id + 1] = temp;
		}
	}
}
