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

namespace IMAC
{

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

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
// ==================================================

	texture<uchar4, 2, cudaReadModeElementType> texInputImg;

	// 15 = taille max de matrice de convolution
	__constant__ float dev_inputMatConv[15*15];

	__device__ float clampfCuda(const float val, const float min , const float max) 
	{
		return fmin(max, fmax(min, val));
	}

	// Fonction qui peut etre appelee a l'interieur d'un global
	// __device__ 
	__global__ void convolution(uchar4 * dev_output,
								const uint imgWidth, const uint imgHeight,
								const uint matSize)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		const int nbThreadsGlobal_x = gridDim.x * blockDim.x; // nb threads global
		const int nbThreadsGlobal_y = gridDim.y * blockDim.y; // nb threads global

		
		uchar4 img;
		while(x < imgWidth && y < imgHeight)
		{
			float3 sum = make_float3(0.f,0.f,0.f);

			// Apply convolution
			for ( uint j = 0; j < matSize; ++j ) 
			{
				for ( uint i = 0; i < matSize; ++i ) 
				{
					int dX = x + i - matSize / 2;
					int dY = y + j - matSize / 2;

					// Handle borders
					if ( dX < 0 ) 
						dX = 0;

					if ( dX >= imgWidth ) 
						dX = imgWidth - 1;

					if ( dY < 0 ) 
						dY = 0;

					if ( dY >= imgHeight ) 
						dY = imgHeight - 1;

					const int idMat		= j * matSize + i;
					// const int idPixel	= dY * imgWidth + dX;

					img = tex2D(texInputImg, dX + 0.5f, dY + 0.5f);
					sum.x += (float)img.x * dev_inputMatConv[idMat];
					sum.y += (float)img.y * dev_inputMatConv[idMat];
					sum.z += (float)img.z * dev_inputMatConv[idMat];
				}
			}		
			const int idOut = y * imgWidth + x;
			dev_output[idOut].x = (uchar)clampfCuda( sum.x, 0.f, 255.f );
			dev_output[idOut].y = (uchar)clampfCuda( sum.y, 0.f, 255.f );
			dev_output[idOut].z = (uchar)clampfCuda( sum.z, 0.f, 255.f );
			dev_output[idOut].w = 255;

			x += nbThreadsGlobal_x;
			y += nbThreadsGlobal_y;
		}
	}

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar4 *dev_inputImg = NULL;
		uchar4 *dev_output = NULL;
		size_t srcPitch;
		
		/// TODOOOOOOOOOOOOOO
		cudaDeviceProp prop;
		cudaGetDeviceProperties( &prop, 0);
		int gridSize_x = sqrt(prop.maxThreadsPerBlock);
		int gridSize_y = sqrt(prop.maxThreadsPerBlock);
		int blockSize_x = (imgWidth + gridSize_x - 1) / gridSize_x;
		int blockSize_y = (imgHeight + gridSize_y - 1) / gridSize_y;

		std::cout << "gridSize " << gridSize_x << " " << gridSize_y << std::endl;
		std::cout << "blockSize " << blockSize_x << " " << blockSize_y << std::endl;
		

		// Allocate arrays on device (input and ouput)
		const size_t bytesImg = inputImg.size() * sizeof(inputImg[0]);
		std::cout 	<< "Allocating input (2 arrays): " 
					<< ( ( 2 * bytesImg ) >> 20 ) << " MB on Device" << std::endl;
		const size_t bytesMat = 15 * 15 * sizeof(float);
		std::cout 	<< "Allocating input (1 array1): " 
					<< ( ( 1 * bytesMat ) >> 20 ) << " MB on Device" << std::endl;

		chrGPU.start();
				
		cudaMallocPitch((void**) &dev_inputImg, &srcPitch, imgWidth * sizeof(inputImg[0]), imgHeight);
		cudaMalloc((void**) &dev_output, bytesImg);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (allocation) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy2D(dev_inputImg, srcPitch, inputImg.data(), imgWidth * sizeof(inputImg[0]), imgWidth * sizeof(inputImg[0]), imgHeight, cudaMemcpyHostToDevice);
		cudaError_t status = cudaBindTexture2D(NULL, texInputImg, dev_inputImg, cudaCreateChannelDesc<uchar4>(), imgWidth, imgHeight, srcPitch);
		if(status != cudaSuccess) {
			std::cout << "ERROR on cudaBindTexture2D : " << cudaGetErrorString(status) << std::endl;
			return;
		}
		
		cudaMemcpyToSymbol(dev_inputMatConv, matConv.data(), bytesMat, 0, cudaMemcpyHostToDevice);

		chrGPU.start();
		
		// Launch kernel
		convolution<<< dim3(blockSize_x, blockSize_y), dim3(gridSize_x, gridSize_y) >>>(dev_output, imgWidth, imgHeight, matSize);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (calcul) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)  
		cudaMemcpy(&output[0], dev_output, bytesImg, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_inputImg);
		cudaUnbindTexture(texInputImg);
		cudaFree(dev_output);

		std::cout << "Checking result..." << std::endl;
		compareImages(resultCPU, output);
	}
}
