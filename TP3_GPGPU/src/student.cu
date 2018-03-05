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
	// ==================================================== Ex 0
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint dev_max[];
		for(int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
			global_idx < size;
			global_idx += blockDim.x * gridDim.x) {

			int local_idx = threadIdx.x;
			//int global_idx = local_idx + blockIdx.x * blockDim.x; // Id global du thread
			dev_max[local_idx] = dev_array[global_idx];
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

		std::cout << "========== Ex 3 " << std::endl;
		/// TODO
		
		std::cout << "========== Ex 4 " << std::endl;
		/// TODO
		
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
