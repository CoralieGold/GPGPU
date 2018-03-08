/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#define USE_CONSTANT
#define USE_TEXTURE_1D
#define USE_TEXTURE_2D

namespace IMAC
{
	// For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	// __global__
 //    void convCUDA(	const uchar4 *const input, const uint imgWidth, const uint imgHeight, 
	// 				const float *const matConv, const uint matSize,
	// 				uchar4 *const output)
	// {
	// 	for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
	// 	{
	// 		for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
	// 		{
	// 			float3 sum = make_float3(0.f,0.f,0.f);
	// 			for (uint j = 0; j < matSize; ++j) 
	// 			{
	// 				for (uint i = 0; i < matSize; ++i) 
	// 				{
	// 					int dX = cudaClampi(x + i - matSize / 2, 0, imgWidth - 1);
	// 					int dY = cudaClampi(y + j - matSize / 2, 0, imgHeight - 1);

	// 					const uint idMat	= j * matSize + i;
	// 					const uint idPixel	= dY * imgWidth + dX;
	// 					sum.x += (float)input[idPixel].x * matConv[idMat];
	// 					sum.y += (float)input[idPixel].y * matConv[idMat];
	// 					sum.z += (float)input[idPixel].z * matConv[idMat];
	// 				}
	// 			}
	// 			const uint idOut = y * imgWidth + x;
	// 			output[idOut].x = (uchar)cudaClampf(sum.x, 0.f, 255.f);
	// 			output[idOut].y = (uchar)cudaClampf(sum.y, 0.f, 255.f);
	// 			output[idOut].z = (uchar)cudaClampf(sum.z, 0.f, 255.f);
	// 			output[idOut].w = 255;
	// 		}
	// 	}
	// }
	
	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
	
    void studentJob(const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytesImg = input.size() * sizeof(uchar4);
		std::cout 	<< "Allocating input, output and convolution matrix on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));

		// Configure kernel
		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
										
		std::cout << "============================ Naive" << std::endl;
		chrGPU.start();
		// convCUDA<<< nbBlocks, nbThreads >>>(dev_input, imgWidth, imgHeight, matSize, dev_output);
		chrGPU.stop();
		std::cout 	<< "-> Done (naive): " << chrGPU.elapsedTime() << " ms" << std::endl;
		
		std::cout << "Checking result..." << std::endl;
		// Copy data from device to host (output array)   
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost)); 
		// Reset dev_output
		HANDLE_ERROR(cudaMemset(dev_output, 0, bytesImg));
		compareImages(resultCPU, output);

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}
