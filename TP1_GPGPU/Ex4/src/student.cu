/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 4: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#define min(a,b) ((a) < (b) ? (a) : (b))

namespace IMAC
{
	__global__ void addMatrices(int* dev_A, int* dev_B, int width, int height, int* dev_output)
	{
		// 2 dimensions : donc recuperation des threads selon les axes x et y
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int x = blockIdx.x * blockDim.x + threadIdx.x;

		const int nbThreadsGlobal_x = gridDim.x * blockDim.x; // Nombre global de threads pour x
		const int nbThreadsGlobal_y = gridDim.y * blockDim.y; // Nombre global de threads pour y

		int idx;

		while(x < width && y < height)
		{
			idx = x + y * width;
			dev_output[idx] = dev_A[idx] + dev_B[idx];

			// Passage a la grid suivante
			x += nbThreadsGlobal_x;
			y += nbThreadsGlobal_y;
		}
	}

	void studentJob(int* a, int* b, int width, int height, int* output)
	{
		// Get Cuda device properties of first device (to dynamically have max number of threads)
		cudaDeviceProp prop;
		cudaGetDeviceProperties( &prop, 0);

		ChronoGPU chrGPU;

		// 2 arrays for GPU
		int* dev_a = NULL;
		int* dev_b = NULL;
		int* dev_output = NULL;

		// On peut avoir au maximum maxThreadsPerBlock, on fait donc un sqrt pour ne pas avoir plus de threads que possible
		int gridSize_x = sqrt(prop.maxThreadsPerBlock);
		int gridSize_y = sqrt(prop.maxThreadsPerBlock);
		int blockSize_x = (width + gridSize_x - 1) / gridSize_x;
		int blockSize_y = (height + gridSize_y - 1) / gridSize_y;

		std::cout << "gridSize " << gridSize_x << " " << gridSize_y << std::endl;
		std::cout << "blockSize " << blockSize_x << " " << blockSize_y << std::endl;
		
		// Allocate arrays on device (input and ouput)
		const size_t bytes = width * height * sizeof(int);
		std::cout 	<< "Allocating input (3 array): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;		
		chrGPU.start();
				
		cudaMalloc((void**) &dev_a, bytes);
		cudaMalloc((void**) &dev_b, bytes);
		cudaMalloc((void**) &dev_output, bytes);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (allocation) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

		chrGPU.start();
		
		// Launch kernel
		addMatrices<<< dim3(blockSize_x, blockSize_y), dim3(gridSize_x, gridSize_y) >>>(dev_a, dev_b, width, height, dev_output);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (calcul) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)  
		cudaMemcpy(&output[0], dev_output, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_output);
	}
}
