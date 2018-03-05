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

namespace IMAC
{
	// Similar to exercise 2, we need to loop if the total number of threads is too small
	// But here, we have a 2D grid... So 2 loops !
	__global__
    void sepiaCUDA(const uchar *const input, const uint width, const uint height, uchar *const output)
	{
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x) 
			{
				const int idPixel = 3 * (x + y * width);
	
				const uchar inRed   = input[idPixel];
				const uchar inGreen = input[idPixel + 1];
				const uchar inBlue  = input[idPixel + 2];
	
				const uchar outRed   = fminf( 255.f, ( inRed * .393f + inGreen * .769f + inBlue * .189f ) );
				const uchar outGreen = fminf( 255.f, ( inRed * .349f + inGreen * .686f + inBlue * .168f ) );
				const uchar outBlue  = fminf( 255.f, ( inRed * .272f + inGreen * .534f + inBlue * .131f ) );
	
				output[idPixel]		= outRed;
				output[idPixel + 1] = outGreen;
				output[idPixel + 2] = outBlue;
			}
		}
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = input.size() * sizeof(uchar);
		std::cout 	<< "Allocating input (2 arrays): " 
					<< ( ( 2 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();		
		HANDLE_ERROR( cudaMalloc( (void**)&dev_input, bytes ) );
		HANDLE_ERROR( cudaMalloc( (void**)&dev_output, bytes ) );		
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input array) 
		std::cout << "Copy data from host to device" << std::endl;
		chrGPU.start();		
		HANDLE_ERROR( cudaMemcpy( dev_input, input.data(), bytes, cudaMemcpyHostToDevice ) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Configure kernel	
		// As asked in the exercise, we use a 2D grid	
		// We should (must) verify if we don't exceed the maximum ! Use cudaGetDeviceProperties ! ;-)
		const dim3 threads(32, 32);
		const dim3 blocks( ( width + threads.x - 1 ) / threads.x, ( height + threads.y - 1 ) / threads.y );

		// Launch kernel
		chrGPU.start();
		sepiaCUDA<<<blocks, threads>>>(dev_input, width, height, dev_output);
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)   
		std::cout << "Copy data from device to host" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy( output.data(), dev_output, bytes, cudaMemcpyDeviceToHost ) ); 
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}
