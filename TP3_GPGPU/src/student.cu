/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	// ==================================================== Ex 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		// Mémoire partagée
		extern __shared__ uint dev_max[];

		// Identifiants du thread en local et global
		int local_idx = threadIdx.x;
		int global_idx = local_idx + blockIdx.x * blockDim.x;

		// Eviter les erreurs d'accès mémoire
		if(global_idx < size)
			// Remplissage du tableau partagé
			dev_max[local_idx] = dev_array[global_idx];
		else
			dev_max[local_idx] = 0;
		__syncthreads();


		for(unsigned int stage = 1; stage < blockDim.x; stage *= 2)  {
			int index = 2 * stage * local_idx;
			if(index < blockDim.x) {
				// Stockage de la valeur max
				dev_max[index] = umax(dev_max[index], dev_max[index + stage]);
			}
			__syncthreads();
		}

		// Garder la valeur max en résultat
		if(local_idx == 0) {
			dev_partialMax[blockIdx.x] = dev_max[0];
		}
	}

	// ==================================================== Ex 2
    __global__
    void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint dev_max[];
		int local_idx = threadIdx.x;
		int global_idx = local_idx + blockIdx.x * blockDim.x;
		if(global_idx < size)
			dev_max[local_idx] = dev_array[global_idx];
		else
			dev_max[local_idx] = 0;
		__syncthreads();

		// Eviter les conflits de banque
		for(unsigned int stage = blockDim.x / 2; stage > 0; stage >>= 1)  {
			if(local_idx < stage) {
				dev_max[local_idx] = umax(dev_max[local_idx], dev_max[local_idx + stage]);
			}
			__syncthreads();
		}

		if(local_idx == 0) {
			dev_partialMax[blockIdx.x] = dev_max[0];
		}
	}

	// ==================================================== Ex 3
    __global__
    void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint dev_max[];
		int local_idx = threadIdx.x;
		int global_idx = local_idx + blockIdx.x * (blockDim.x * 2);
		if(global_idx < size)
			// Maximisation de l'activité des threads
			dev_max[local_idx] = umax(dev_array[global_idx], dev_max[global_idx + blockDim.x]);
		else
			dev_max[local_idx] = 0;
		__syncthreads();

		for(unsigned int stage = blockDim.x / 2; stage > 0; stage >>= 1)  {
			if(local_idx < stage) {
				dev_max[local_idx] = umax(dev_max[local_idx], dev_max[local_idx + stage]);
			}
			__syncthreads();
		}

		if(local_idx == 0) {
			dev_partialMax[blockIdx.x] = dev_max[0];
		}
	}

	// ==================================================== Ex 4
    __global__
    void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint dev_max[];
		int local_idx = threadIdx.x;
		int global_idx = local_idx + blockIdx.x * (blockDim.x * 2);
		if(global_idx < size)
			dev_max[local_idx] = umax(dev_array[global_idx + blockDim.x], dev_max[global_idx + blockDim.x + (blockDim.x/2)]);
		else
			dev_max[local_idx] = 0;
		__syncthreads();

		for(unsigned int stage = blockDim.x / 4; stage > 32; stage >>= 1)  {
			if(local_idx < stage) {
				dev_max[local_idx] = umax(dev_max[local_idx], dev_max[local_idx + stage]);
			}
			__syncthreads();
		}

		// Déroulé le wrap
		if (local_idx < 32) {
			warpReduce(dev_max, local_idx);
		}
	}

	__device__ void warpReduce(volatile uint* dev_max, int local_idx) {

		dev_max[local_idx]+=dev_max[local_idx+32];
		dev_max[local_idx]+=dev_max[local_idx+16];
		dev_max[local_idx]+=dev_max[local_idx+8];
		dev_max[local_idx]+=dev_max[local_idx+4];
		dev_max[local_idx]+=dev_max[local_idx+2];
		dev_max[local_idx]+=dev_max[local_idx+1];
	}


	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(dev_array, array.size(), res1);
		
        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		
		std::cout << "========== Ex 2 " << std::endl;
		/// TODO
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(dev_array, array.size(), res2);
		
        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU); // Compare results

		std::cout << "========== Ex 3 " << std::endl;
		/// TODO
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX2>(dev_array, array.size(), res3);
	
        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU); // Compare results
	
		std::cout << "========== Ex 4 " << std::endl;
		/// TODO
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX2>(dev_array, array.size(), res4);
	
        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU); // Compare results
		
		std::cout << "========== Ex 5 " << std::endl;
		/// TODO
		

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
