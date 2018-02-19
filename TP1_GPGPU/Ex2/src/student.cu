/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x; // Id global du thread
		const int nbThreadsGlobal = gridDim.x * blockDim.x; // Nombre de threads global

		while(idx < n)
		{
			// Somme des tableaux
			dev_res[idx] = dev_a[idx] + dev_b[idx];

			// Passage a la grid suivante
			idx += nbThreadsGlobal;
		}
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		// Get Cuda device properties of first device (to dynamically have max number of threads)
		cudaDeviceProp prop;
		cudaGetDeviceProperties( &prop, 0);

		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;		
		chrGPU.start();

		cudaMalloc((void**) &dev_a, bytes);
		cudaMalloc((void**) &dev_b, bytes);
		cudaMalloc((void**) &dev_res, bytes);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (allocation) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

		chrGPU.start();
		
		// Get maxThreadsPerBlock and number of blocks
		int nbThreads = prop.maxThreadsPerBlock;
		int nbBlocks = (size + nbThreads - 1) / nbThreads;

		std::cout << "nbBlocks = " << nbBlocks << " | nbThreads = " << nbThreads << std::endl;

		// Launch kernel
		sumArraysCUDA<<< nbBlocks, nbThreads >>>(size, dev_a, dev_b, dev_res);
		chrGPU.stop();
		std::cout 	<< "-> Done (calcul) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)  
		cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);
		
		// Free arrays on device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
	}
}

