
#include <iostream>
#include <stdlib.h>

#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 10

__global__ void sort(int *d_in, int *d_out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, key, j;
	for (i = 1; i < N; i++)
	{
		key = d_in[i];
		j = i - 1;

		/* Move elements of arr[0..i-1], that are
		greater than key, to one position ahead
		of their current position */
		while (j >= 0 && d_in[j] > key)
		{
			d_in[j + 1] = d_in[j];
			j = j - 1;
		}
		d_in[j + 1] = key;
	}
}


int main()
{
	int h_in[N] = {0}, h_out[N] = {0};

	for (int i = 0; i < N; i++)
	{
		h_in[i] = rand() % 100;
	}
	
	int *d_in, *d_out;
	cudaMalloc(&d_in, N * sizeof(int));
	cudaMalloc(&d_out, N * sizeof(int));

	cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	sort <<< numBlocks, blockSize >>> (d_in, d_out);

	cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		printf("%d\t", h_in[i]);
	}
	printf("\n");
	for (int i = 0; i < N; i++)
	{
		printf("%d\t", h_out[i]);
	}
	printf("\n");
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}