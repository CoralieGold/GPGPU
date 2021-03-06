/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.hpp
* Author: Maxime MARIA
*/

#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"
#include "chronoGPU.hpp"
#include "chronoCPU.hpp"

namespace IMAC
{
	const uint MAX_NB_THREADS = 1024; // En dur, changer si GPU plus ancien ;-)
    const uint DEFAULT_NB_BLOCKS = 32768;

    enum
    {
        KERNEL_EX1 = 0,
        KERNEL_EX2,
        KERNEL_EX3,
        KERNEL_EX4,
        KERNEL_EX5
    };

    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax);
    __global__
    void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax);
    __global__
    void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax);
    __global__
    void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax);
 	__device__ void warpReduce(volatile uint* dev_max, int local_idx);

    template<uint kernelType>
    uint2 configureKernel(const uint sizeArray)
    {
        uint2 dimBlockGrid; // x: dimBlock / y: dimGrid

		// Configure number of threads/blocks
		switch(kernelType)
		{
			case KERNEL_EX1:
				dimBlockGrid.x = MAX_NB_THREADS; 
				// Faire en sorte que ça fonctionne pour toutes les tailles de tableau
				dimBlockGrid.y = (sizeArray + MAX_NB_THREADS - 1) / MAX_NB_THREADS;
			break;
			case KERNEL_EX2:
				/// TODO EX 2
				dimBlockGrid.x = MAX_NB_THREADS; 
				dimBlockGrid.y = (sizeArray + MAX_NB_THREADS - 1) / MAX_NB_THREADS;
			break;
			case KERNEL_EX3:
				/// TODO EX 3
				dimBlockGrid.x = MAX_NB_THREADS; 
				dimBlockGrid.y = (sizeArray + MAX_NB_THREADS - 1) / MAX_NB_THREADS;
			break;
			case KERNEL_EX4:
				/// TODO EX 4
				dimBlockGrid.x = MAX_NB_THREADS; 
				dimBlockGrid.y = (sizeArray + MAX_NB_THREADS - 1) / MAX_NB_THREADS;
			break;
			case KERNEL_EX5:
				/// TODO EX 5
			break;
            default:
                throw std::runtime_error("Error configureKernel: unknown kernel type");
		}
		verifyDimGridBlock( dimBlockGrid.y, dimBlockGrid.x, sizeArray ); // Are you reasonable ?
        
        return dimBlockGrid;
    }

    // Launch kernel number 'kernelType' and return float2 for timing (x:device,y:host)    
    template<uint kernelType>
    float2 reduce(const uint *const dev_array, const uint size, uint &result)
	{
        const uint2 dimBlockGrid = configureKernel<kernelType>(size);

		// Allocate arrays (host and device) for partial result
		/// TODO
		// Taille de la mémoire partagée
		const size_t sizeSharedMemory = dimBlockGrid.x * sizeof(uint);

		// On a besoin d'avoir comme taille le nombre de blocks 
		std::vector<uint> host_partialMax(dimBlockGrid.y);

		uint *dev_partialMax;
		const size_t bytesPartialMax = dimBlockGrid.y * sizeof(uint); 

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_partialMax, bytesPartialMax ) );

		ChronoGPU chrGPU;
		float2 timing; // x: timing GPU, y: timing CPU
		const uint loop = 100;
		// Average timing on 'loop' iterations
		chrGPU.start();
		for (uint i = 0; i < loop; ++i)
		{
			switch(kernelType) // Template : evaluation at compilation time ! ;-)
			{
				case KERNEL_EX1:
					/// TODO EX 1 
					maxReduce_ex1<<< dimBlockGrid.y, dimBlockGrid.x, sizeSharedMemory >>>(dev_array, size, dev_partialMax);
				break;
				case KERNEL_EX2:
					/// TODO EX 2
					maxReduce_ex2<<< dimBlockGrid.y, dimBlockGrid.x, sizeSharedMemory >>>(dev_array, size, dev_partialMax);
				break;
				case KERNEL_EX3:
					/// TODO EX 3
					maxReduce_ex3<<< dimBlockGrid.y, dimBlockGrid.x, sizeSharedMemory >>>(dev_array, size, dev_partialMax);
				break;
				case KERNEL_EX4:
					/// TODO EX 4
					maxReduce_ex4<<< dimBlockGrid.y, dimBlockGrid.x, sizeSharedMemory >>>(dev_array, size, dev_partialMax);
				break;
				case KERNEL_EX5:
					/// TODO EX 5
				break;
                default:
		            cudaFree(dev_partialMax);
                    throw("Error reduce: unknown kernel type.");
			}
		}
		chrGPU.stop();
		timing.x = chrGPU.elapsedTime() / (float)loop; // Stores time for device

		// Retrieve partial result from device to host
		HANDLE_ERROR(cudaMemcpy(host_partialMax.data(), dev_partialMax, bytesPartialMax, cudaMemcpyDeviceToHost));


		cudaFree(dev_partialMax);

        // Check for error
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			throw std::runtime_error(cudaGetErrorString(err));
		}
		
		ChronoCPU chrCPU;
		chrCPU.start();

		// Finish on host
		for (int i = 0; i < host_partialMax.size(); ++i)
		{
			result = std::max<uint>(result, host_partialMax[i]);
		}
		
		chrCPU.stop();

		timing.y = chrCPU.elapsedTime(); // Stores time for host
        return timing;
	}  
    
    void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */);

    void printTiming(const float2 timing);
    void compare(const uint resGPU, const uint resCPU);
}

#endif
