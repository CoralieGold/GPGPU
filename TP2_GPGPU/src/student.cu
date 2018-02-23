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

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

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

	// Fonction clamp mais avec les fonctions de min et max pour Cuda
	__device__ float clampfCuda(const float val, const float min , const float max) 
	{
		return fmin(max, fmax(min, val));
	}

// ==================================================

	__global__ void convolution1(const uchar4 *const dev_inputImg,
								const float *const dev_matConv,
								uchar4 * dev_output,
								const uint imgWidth, const uint imgHeight,
								const uint matSize)
	{
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
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
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)dev_inputImg[idPixel].x * dev_matConv[idMat];
						sum.y += (float)dev_inputImg[idPixel].y * dev_matConv[idMat];
						sum.z += (float)dev_inputImg[idPixel].z * dev_matConv[idMat];
					}
				}		
				const int idOut = y * imgWidth + x;
				dev_output[idOut].x = (uchar)clampfCuda( sum.x, 0.f, 255.f );
				dev_output[idOut].y = (uchar)clampfCuda( sum.y, 0.f, 255.f );
				dev_output[idOut].z = (uchar)clampfCuda( sum.z, 0.f, 255.f );
				dev_output[idOut].w = 255;

			}
		}
	}

	void studentJob1(const std::vector<uchar4> &inputImg, // Input image
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
		float *dev_inputMatConv = NULL;
		uchar4 *dev_output = NULL;
		
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
		const size_t bytesImg = inputImg.size() * sizeof(uchar4);
		const size_t bytesMat = matConv.size() * sizeof(float);
		std::cout 	<< "Allocating input (2 arrays): " 
					<< ( ( bytesImg + bytesMat ) >> 20 ) << " MB on Device" << std::endl;
		std::cout 	<< "Allocating output (1 array): " 
					<< ( ( bytesImg ) >> 20 ) << " MB on Device" << std::endl;				
		chrGPU.start();
				
		HANDLE_ERROR( cudaMalloc((void**) &dev_inputImg, bytesImg) );
		HANDLE_ERROR( cudaMalloc((void**) &dev_inputMatConv, bytesMat) );
		HANDLE_ERROR( cudaMalloc((void**) &dev_output, bytesImg) );
		
		chrGPU.stop();
		std::cout 	<< "-> Done (allocation) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_inputImg, &inputImg[0], bytesImg, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_inputMatConv, &matConv[0], bytesMat, cudaMemcpyHostToDevice);

		chrGPU.start();
		
		// Launch kernel
		convolution1<<< dim3(blockSize_x, blockSize_y), dim3(gridSize_x, gridSize_y) >>>(dev_inputImg, dev_inputMatConv, dev_output, imgWidth, imgHeight, matSize);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (calcul) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)  
		HANDLE_ERROR( cudaMemcpy(&output[0], dev_output, bytesImg, cudaMemcpyDeviceToHost) );

		// Free arrays on device
		cudaFree(dev_inputImg);
		cudaFree(dev_inputMatConv);
		cudaFree(dev_output);

		std::cout << "Checking result..." << std::endl;
		compareImages(resultCPU, output);
	}


// ==================================================

	// Declaration d'une variable constante: 
	// Tableau de foat de taille 15*15
	// 15 = taille max pour la matrice de convolution
	__constant__ float dev_inputMatConv[15*15];

	__global__ void convolution2(const uchar4 *const dev_inputImg,
								uchar4 * dev_output,
								const uint imgWidth, const uint imgHeight,
								const uint matSize)
	{
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
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
						const int idPixel	= dY * imgWidth + dX;

						// Pour les variables constantes, on recupere les donnees de la meme maniere qu'avant
						sum.x += (float)dev_inputImg[idPixel].x * dev_inputMatConv[idMat];
						sum.y += (float)dev_inputImg[idPixel].y * dev_inputMatConv[idMat];
						sum.z += (float)dev_inputImg[idPixel].z * dev_inputMatConv[idMat];
					}
				}		
				const int idOut = y * imgWidth + x;
				dev_output[idOut].x = (uchar)clampfCuda( sum.x, 0.f, 255.f );
				dev_output[idOut].y = (uchar)clampfCuda( sum.y, 0.f, 255.f );
				dev_output[idOut].z = (uchar)clampfCuda( sum.z, 0.f, 255.f );
				dev_output[idOut].w = 255;
			}
		}
	}

	void studentJob2(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar4 *dev_inputImg = NULL;
		uchar4 *dev_output = NULL;
		
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
		const size_t bytesImg = inputImg.size() * sizeof(uchar4);
		const size_t bytesMat = 15 * 15 * sizeof(float);
		std::cout 	<< "Allocating input (2 arrays): " 
					<< ( ( bytesImg + bytesMat) >> 20 ) << " MB on Device" << std::endl;
		
		std::cout 	<< "Allocating output (1 array1): " 
					<< ( ( bytesImg ) >> 20 ) << " MB on Device" << std::endl;				
		chrGPU.start();
				
		cudaMalloc((void**) &dev_inputImg, bytesImg);
		cudaMalloc((void**) &dev_output, bytesImg);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (allocation) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_inputImg, &inputImg[0], bytesImg, cudaMemcpyHostToDevice);
		// CudaMemcpy mais pour une variable __constant__ cuda
		cudaMemcpyToSymbol(dev_inputMatConv, matConv.data(), bytesMat, 0, cudaMemcpyHostToDevice);

		chrGPU.start();
		
		// Launch kernel
		convolution2<<< dim3(blockSize_x, blockSize_y), dim3(gridSize_x, gridSize_y) >>>(dev_inputImg, dev_output, imgWidth, imgHeight, matSize);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (calcul) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)  
		cudaMemcpy(&output[0], dev_output, bytesImg, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_inputImg);
		cudaFree(dev_output);

		std::cout << "Checking result..." << std::endl;
		compareImages(resultCPU, output);
	}

// ==================================================

	// Declaration d'une texture 1D de uchar4
	texture<uchar4, 1, cudaReadModeElementType> texInputImg;

	__global__ void convolution3(uchar4 * dev_output,
								const uint imgWidth, const uint imgHeight,
								const uint matSize)
	{
		uchar4 img;

		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
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
						const int idPixel	= dY * imgWidth + dX;

						// On utilise tex1Dfetch pour récupérer la texture à un pixel précis
						img = tex1Dfetch(texInputImg, idPixel);
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
			}
		}
	}

    void studentJob3(const std::vector<uchar4> &inputImg, // Input image
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
		const size_t bytesImg = inputImg.size() * sizeof(uchar4);
		std::cout 	<< "Allocating input (2 arrays): " 
					<< ( ( 2 * bytesImg ) >> 20 ) << " MB on Device" << std::endl;
		const size_t bytesMat = 15 * 15 * sizeof(float);
		std::cout 	<< "Allocating input (1 array1): " 
					<< ( ( 1 * bytesMat ) >> 20 ) << " MB on Device" << std::endl;				
		chrGPU.start();
				
		cudaMalloc((void**) &dev_inputImg, bytesImg);
		cudaMalloc((void**) &dev_output, bytesImg);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (allocation) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_inputImg, &inputImg[0], bytesImg, cudaMemcpyHostToDevice);
		cudaBindTexture(NULL, texInputImg, dev_inputImg, bytesImg);
		cudaMemcpyToSymbol(dev_inputMatConv, matConv.data(), bytesMat, 0, cudaMemcpyHostToDevice);

		chrGPU.start();
		
		// Launch kernel
		convolution3<<< dim3(blockSize_x, blockSize_y), dim3(gridSize_x, gridSize_y) >>>(dev_output, imgWidth, imgHeight, matSize);
		
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


// ==================================================

	// Declaration d'une texture 2D de uchar4
	texture<uchar4, 2, cudaReadModeElementType> texInputImg2D;

	__global__ void convolution4(uchar4 * dev_output,
								const uint imgWidth, const uint imgHeight,
								const uint matSize)
	{
		uchar4 img;
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
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
						
						// En 2D, tex1Dfetch est remplace par tex2D
						// On ajoute 0.5 aux coordonnees pour bien obtenir le centre du pixel
						img = tex2D(texInputImg2D, dX + 0.5f, dY + 0.5f);
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
			}
		}
	}

	void studentJob4(const std::vector<uchar4> &inputImg, // Input image
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
		std::cout 	<< "Allocating output (1 array): " 
					<< ( ( bytesImg ) >> 20 ) << " MB on Device" << std::endl;
		const size_t bytesMat = 15 * 15 * sizeof(float);
		std::cout 	<< "Allocating input (1 array): " 
					<< ( ( bytesMat ) >> 20 ) << " MB on Device" << std::endl;

		chrGPU.start();
				
		cudaMallocPitch((void**) &dev_inputImg, &srcPitch, imgWidth * sizeof(inputImg[0]), imgHeight);
		cudaMalloc((void**) &dev_output, bytesImg);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (allocation) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy2D(dev_inputImg, srcPitch, inputImg.data(), imgWidth * sizeof(inputImg[0]), imgWidth * sizeof(inputImg[0]), imgHeight, cudaMemcpyHostToDevice);
		cudaError_t status = cudaBindTexture2D(NULL, texInputImg2D, dev_inputImg, cudaCreateChannelDesc<uchar4>(), imgWidth, imgHeight, srcPitch);
		if(status != cudaSuccess) {
			std::cout << "ERROR on cudaBindTexture2D : " << cudaGetErrorString(status) << std::endl;
			return;
		}
		
		cudaMemcpyToSymbol(dev_inputMatConv, matConv.data(), bytesMat, 0, cudaMemcpyHostToDevice);

		chrGPU.start();
		
		// Launch kernel
		convolution4<<< dim3(blockSize_x, blockSize_y), dim3(gridSize_x, gridSize_y) >>>(dev_output, imgWidth, imgHeight, matSize);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (calcul) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)  
		cudaMemcpy(&output[0], dev_output, bytesImg, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_inputImg);
		cudaUnbindTexture(texInputImg2D);
		cudaFree(dev_output);

		std::cout << "Checking result..." << std::endl;
		compareImages(resultCPU, output);
	}

// ==================================================

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		std::cout << " ------------ EXERCICE 1 ------------ " << std::endl;
		studentJob1(inputImg, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
		
		std::cout << "\n ------------------------------------ " << std::endl;
		std::cout << " ------------ EXERCICE 2 ------------ " << std::endl;
		studentJob2(inputImg, imgWidth, imgHeight, matConv, matSize, resultCPU, output);

		std::cout << "\n ------------------------------------ " << std::endl;
		std::cout << " ------------ EXERCICE 3 ------------ " << std::endl;
		studentJob3(inputImg, imgWidth, imgHeight, matConv, matSize, resultCPU, output);

		std::cout << "\n ------------------------------------ " << std::endl;
		std::cout << " ------------ EXERCICE 4 ------------ " << std::endl;
		studentJob4(inputImg, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
	
	}
}
