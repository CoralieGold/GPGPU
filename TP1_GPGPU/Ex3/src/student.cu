/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#define min(a,b) ((a) < (b) ? (a) : (b))

namespace IMAC
{
	__global__ void sepia(const uchar *const dev_input, uchar *const dev_output, const uint width, const uint height)
	{
		// 2 dimensions : donc recuperation des threads selon les axes x et y
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int idx;

		const int nbThreadsGlobal_x = gridDim.x * blockDim.x; // Nombre global de threads pour x
		const int nbThreadsGlobal_y = gridDim.y * blockDim.y; // Nombre global de threads pour y

		while(x < width && y < height)
		{
			// Recuperation du pixel
			idx = (y * width + x) * 3;

			// Pour chaque couleur, calcul du sepia (selon les formules donnees dans l'enonce)
			dev_output[idx] = min(255, dev_input[idx]*0.393 + dev_input[idx + 1]*0.769 + dev_input[idx + 2]*0.189);
			dev_output[idx + 1] = min(255, dev_input[idx]*0.349 + dev_input[idx + 1]*0.686 + dev_input[idx + 2]*0.168);
			dev_output[idx + 2] = min(255, dev_input[idx]*0.272 + dev_input[idx + 1]*0.534 + dev_input[idx + 2]*0.131);
		
			x += nbThreadsGlobal_x;
			y += nbThreadsGlobal_y;
		}
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		// Get Cuda device properties of first device (to dynamically have max number of threads)
		cudaDeviceProp prop;
		cudaGetDeviceProperties( &prop, 0);

		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;
		
		// On peut avoir au maximum maxThreadsPerBlock, on fait donc un sqrt pour ne pas avoir plus de threads que possible
		int gridSize_x = sqrt(prop.maxThreadsPerBlock);
		int gridSize_y = sqrt(prop.maxThreadsPerBlock);
		int blockSize_x = (width + gridSize_x - 1) / gridSize_x;
		int blockSize_y = (height + gridSize_y - 1) / gridSize_y;

		std::cout << "gridSize " << gridSize_x << " " << gridSize_y << std::endl;
		std::cout << "blockSize " << blockSize_x << " " << blockSize_y << std::endl;
		
		// Allocate arrays on device (input and ouput)
		const size_t bytes = input.size() * sizeof(uchar);
		std::cout 	<< "Allocating input (2 arrays): " 
					<< ( ( 2 * bytes ) >> 20 ) << " MB on Device" << std::endl;		
		chrGPU.start();
				
		cudaMalloc((void**) &dev_input, bytes);
		cudaMalloc((void**) &dev_output, bytes);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (allocation) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_input, &input[0], bytes, cudaMemcpyHostToDevice);

		chrGPU.start();
		
		// Launch kernel
		sepia<<< dim3(blockSize_x, blockSize_y), dim3(gridSize_x, gridSize_y) >>>(dev_input, dev_output, width, height);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (calcul) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)  
		cudaMemcpy(&output[0], dev_output, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}
