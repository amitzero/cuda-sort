#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

/**********************************************************
 * **********************************************************
 *	error checking stufff
 ***********************************************************
 ***********************************************************/
// Enable this for error checking
#define CUDA_CHECK_ERROR

#define CudaSafeCall(err) cudaSafeCall(err, FILE, LINE)
#define CudaCheckError() cudaCheckError(FILE, LINE)

inline void cudaSafeCall(cudaError_t err, const char *file, const int line)
{
#ifdef CUDA_CHECK_ERROR

#pragma warning(push)
#pragma warning(disable : 4127) // Prevent warning on do-while(0);
    do
    {
        if (cudaSuccess != err)
        {
            const char* err_str = cudaGetErrorString(err);
            fprintf(stderr, "At %s:%i : %s cudaSafeCall() failed\n", file, line, err_str);
            exit(-1);
        }
    } while (0);

#pragma warning(pop)
#endif // CUDA_CHECK_ERROR
    return;
}

inline void cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_CHECK_ERROR

#pragma warning(push)
#pragma warning(disable : 4127) // Prevent warning on do-while(0);

    do
    {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            const char* err_str = cudaGetErrorString(err);
            fprintf(stderr, "At %s:%i : %s cudaSafeCall() failed\n", file, line, err_str);
            exit(-1);
        }

        // More careful checking. However, this will affect performance.
        // Comment if not needed.
        err = cudaDeviceSynchronize();
        if (cudaSuccess != err)
        {
            const char* err_str = cudaGetErrorString(err);
            fprintf(stderr, "At %s:%i : %s cudaSafeCall() failed\n", file, line, err_str);
            exit(-1);
        }
    } while (0);

#pragma warning(pop)
#endif // CUDA_CHECK_ERROR
    return;
}

/***************************************************************
 * **************************************************************
 *	end of error checking stuff
 ****************************************************************
 ***************************************************************/

// function takes an array pointer, and the number of rows and cols in the array, and
// allocates and intializes the array to a bunch of random numbers
// Note that this function creates a 1D array that is a flattened 2D array
// to access data item data[i][j], you must can use data[(i*rows) + j]
int *makeRandArray(const int size, const int seed)
{
    srand(seed);
    int *array;
    cudaMalloc((void **)&array, size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

//*******************************//
// your kernel here!!!!!!!!!!!!!!!!!
//*******************************//
__global__ void matavgKernel(int* array, int size)
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


int main(int argc, char *argv[])
{

    int *array; // the poitner to the array of rands int size, seed; // values for the size of the array bool printSorted = false;
    // and the seed for generating
    // random numbers

    // check the command line args
    if (argc < 4)
    {
        std::cerr << "usage: "
                  << argv[0]
                  << " [amount of random nums to generate] [seed value for rand]"
                  << " [1 to print sorted array, 0 otherwise]"
                  << std::endl;
        exit(-1);
    }
    int size = 0;
    // convert cstrings to ints
    {
        std::stringstream ss1(argv[1]);
        ss1 >> size;
    }
    int seed = 0;
    {
        std::stringstream ss1(argv[2]);
        ss1 >> seed;
    }
    bool printSorted = false;
    {
        int sortPrint;
        std::stringstream ss1(argv[3]);
        ss1 >> sortPrint;
        if (sortPrint == 1)
            printSorted = true;
    }
    // get the random numbers
    array = makeRandArray(size, seed);

    /***********************************
     *	create a cuda timer to time execution
     **********************************/
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);
    /***********************************
     *	end of cuda timer creation
     **********************************/

    /////////////////////////////////////////////////////////////////////
    /////////////////////// YOUR CODE HERE	///////////////////////
    /////////////////////////////////////////////////////////////////////

    matavgKernel<<<1,size>>>(array, size);

    /***********************************
     *	Stop and destroy the cuda timer
     **********************************/
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    /***********************************
     *	end of cuda timer destruction
     **********************************/

    std::cerr << "Total time in seconds: "
              << timeTotal / 1000.0 << std::endl;
    if (printSorted)
    {

        ///////////////////////////////////////////////
        /// Your code to print the sorted array here //
        ///////////////////////////////////////////////
        for (int i = 0; i < size; i++)
        {
            std::cout << array[i] << std::endl;
        }
    }
    cudaFree(array);
}
