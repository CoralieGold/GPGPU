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

// #define OPTIM_SHARED_MEM
// #define OPTIM_TEXTURE_1D
#define OPTIM_CONSTANT_MEM

namespace IMAC
{
	// For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar3 &c) {
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "]";  
		return os; 
	}

	__device__ __constant__ uint c_nbLevels;

#ifdef OPTIM_TEXTURE_1D
	texture<uchar3, cudaTextureType1D, cudaReadModeElementType> t_in1D;
#endif

#ifdef OPTIM_CONSTANT_MEM
	__device__ __constant__ int c_histogram[256];
#endif

	__global__ void hsvToRgbCUDA(const float *const hue, const float *const saturation, 
		const float *const value, const uint imgWidth, const uint imgHeight, uchar3 *output) {

		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y)  {
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) {

				const uint idOut = y * imgWidth + x;

				float c = (float)(value[idOut] * saturation[idOut]);

				float x = c * (1.f - fabs(fmod((float)hue[idOut] / 60.f, 2.f) - 1.f));
				float m = (float)(value[idOut] - c);

				float red, green, blue;

				if(hue[idOut] < 60.f) {
					red = c;
					green = x;
					blue = 0.f;
				}
				else if(hue[idOut] >= 60.f && hue[idOut] < 120.f) {
					red = x;
					green = c;
					blue = 0.f;
				}
				else if(hue[idOut] >= 120.f && hue[idOut] < 180.f) {
					red = 0.f;
					green = c;
					blue = x;
				}
				else if(hue[idOut] >= 180.f && hue[idOut] < 240.f) {
					red = 0.f;
					green = x;
					blue = c;
				}
				else if(hue[idOut] >= 240.f && hue[idOut] < 300.f) {
					red = x;
					green = 0.f;
					blue = c;
				}
				else {
					red = c;
					green = 0.f;
					blue = x;
				}

				output[idOut].x = (uchar)((red + m) * 256.f);
				output[idOut].y = (uchar)((green + m) * 256.f);
				output[idOut].z = (uchar)((blue + m) * 256.f);
			}
		}
	}

#ifdef OPTIM_TEXTURE_1D
	__global__ void rgbToHsvCUDA(const uint imgWidth, const uint imgHeight, float *hue, float *saturation, float *value) {

		
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const uint idPixel = x + y * imgWidth;
				uchar3 in = tex1Dfetch<uchar3>(t_in1D, idPixel); // Get data from 1D texture

				float red = (float)in.x / c_nbLevels;
				float green = (float)in.y / c_nbLevels;
				float blue = (float)in.z / c_nbLevels;

				float cMax = fmax(fmax(red, green), blue);
				float cMin = fmin(fmin(red, green), blue);
				float delta = (float)(cMax - cMin);

				hue[idPixel] = 0.f;
				if(cMax == red)        hue[idPixel] = 60.f * fmod(((green - blue) / delta), 6.f);
				else if(cMax == green) hue[idPixel] = 60.f * (((blue - red) / delta) + 2.f);
				else if(cMax == blue)  hue[idPixel] = 60.f * (((red - green) / delta) + 4.f);
				else                   hue[idPixel] = 0.f;
				if(hue[idPixel] < 0.f) hue[idPixel] += 360.f;

				if(cMax != 0.f) saturation[idPixel] = (float)(delta / cMax);
				else saturation[idPixel] = 0.f;

				value[idPixel] = cMax;
			}
		}
	}
#else
	__global__ void rgbToHsvCUDA(const uchar3 *const input, const uint imgWidth, const uint imgHeight, float *hue, float *saturation, float *value) {

		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const int idPixel = x + y * imgWidth;

				float red = (float)input[idPixel].x / c_nbLevels;
				float green = (float)input[idPixel].y / c_nbLevels;
				float blue = (float)input[idPixel].z / c_nbLevels;

				float cMax = fmax(fmax(red, green), blue);
				float cMin = fmin(fmin(red, green), blue);
				float delta = (float)(cMax - cMin);

				hue[idPixel] = 0.f;
				if(cMax == red)        hue[idPixel] = 60.f * fmod(((green - blue) / delta), 6.f);
				else if(cMax == green) hue[idPixel] = 60.f * (((blue - red) / delta) + 2.f);
				else if(cMax == blue)  hue[idPixel] = 60.f * (((red - green) / delta) + 4.f);
				else                   hue[idPixel] = 0.f;
				if(hue[idPixel] < 0.f) hue[idPixel] += 360.f;

				if(cMax != 0.f) saturation[idPixel] = (float)(delta / cMax);
				else saturation[idPixel] = 0.f;

				value[idPixel] = cMax;
			}
		}
	}
#endif

#ifdef OPTIM_CONSTANT_MEM
	__global__ void cumulativeDistributionCUDA(int *repartition) {
		repartition[threadIdx.x] = c_histogram[threadIdx.x];
		__syncthreads();

		int step = 1;
		while(step <= c_nbLevels) {
			int index = (threadIdx.x + 1) * step * 2 - 1;
			if(index < 2 * c_nbLevels) {
				repartition[index] += repartition[index - step];
			}
			step *= 2;
			__syncthreads();
		}

		step = c_nbLevels / 2;
		while(step > 0) {
			int index = (threadIdx.x + 1) * step * 2 - 1;
			if(index < 2 * c_nbLevels) {
				repartition[index + step] += repartition[index];
			}
			step /= 2;
			__syncthreads();
		}
	}
#else
	__global__ void cumulativeDistributionCUDA(const int *const histogram, int *repartition) {
		repartition[threadIdx.x] = histogram[threadIdx.x];
		__syncthreads();

		int step = 1;
		while(step <= c_nbLevels) {
			int index = (threadIdx.x + 1) * step * 2 - 1;
			if(index < 2 * c_nbLevels) {
				repartition[index] += repartition[index - step];
			}
			step *= 2;
			__syncthreads();
		}

		step = c_nbLevels / 2;
		while(step > 0) {
			int index = (threadIdx.x + 1) * step * 2 - 1;
			if(index < 2 * c_nbLevels) {
				repartition[index + step] += repartition[index];
			}
			step /= 2;
			__syncthreads();
		}
	}
#endif

#ifdef OPTIM_SHARED_MEM
	__global__ void histogramCUDA(int *histogram, const float *const value, const uint imgWidth, const uint imgHeight) {
		extern __shared__ int sharedHistogram[];
		for (int i = threadIdx.x; i < c_nbLevels; i += blockDim.x) {
			sharedHistogram[i] = 0;
			__syncthreads();
		}

		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) {
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) { 
				int myId = x + y * imgWidth;
			    int myItem = value[myId]*c_nbLevels;
			    int myBin = myItem % c_nbLevels;
				atomicAdd(&sharedHistogram[myBin], 1);
				
				histogram[myBin] += sharedHistogram[myBin];
				__syncthreads();
			}
		}
	}
#else
	__global__ void histogramCUDA(int *histogram, const float *const value, const uint imgWidth, const uint imgHeight) {
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) {
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) { 
				int myId = x + y * imgWidth;
			    int myItem = value[myId]*c_nbLevels;
			    int myBin = myItem % c_nbLevels;
				atomicAdd(&histogram[myBin], 1);
			}
		}
	}
#endif

	__global__ void equalizationCUDA(float *value, const int *const repartition, const uint imgWidth, const uint imgHeight) {
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const int idPixel = x + y * imgWidth;
			    int myItem = value[idPixel]*256;
			    int myBin = myItem % 256;
				value[idPixel] = (float)(repartition[myBin] - repartition[0])/((imgWidth*imgHeight)-1.f);
			}
		}
	}

	void compareImages(const std::vector<uchar3> &a, const std::vector<uchar3> &b)
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
				if (    std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].z - b[i].z) > 2 )
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

	void studentJob(const std::vector<uchar3> &input, const uint imgWidth, const uint imgHeight, 
					const uint nbLevels,
					const std::vector<uchar3> &resultCPU, // Just for comparison
					std::vector<uchar3> &output) {

		ChronoGPU chrGPU;

		// 7 arrays for GPU
		uchar3 *dev_input = NULL;
		uchar3 *dev_output = NULL;

		float *dev_hue = NULL;
		float *dev_saturation = NULL;
		float *dev_value = NULL;

		int *dev_histogram = NULL;
		int *dev_repartition = NULL;


		/****** Allocate arrays on device (input and ouput) ******/
		const uint imageSize = imgHeight * imgWidth;

		const size_t bytesImg = imageSize * sizeof(uchar3);
		std::cout   << "Allocating input and output images on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));

		const size_t bytesFloatImg = imageSize * sizeof(float);
		std::cout   << "Allocating hue, saturation and value on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_hue, bytesFloatImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_saturation, bytesFloatImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_value, bytesFloatImg));

		const size_t bytesLevel = nbLevels * sizeof(int);
		std::cout   << "Allocating histogram and repartition on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_histogram, bytesLevel));
		HANDLE_ERROR(cudaMemset(dev_histogram, 0, bytesLevel));
		HANDLE_ERROR(cudaMalloc((void**)&dev_repartition, bytesLevel));

		std::cout << "Copy data from host to device" << std::endl;
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), bytesImg, cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpyToSymbol(c_nbLevels, &nbLevels, sizeof(uint)));

#ifdef USE_TEXTURE_1D
		// Bind 1D texture
		HANDLE_ERROR(cudaBindTexture(NULL, t_in1D, dev_input, bytesImg));
#endif


		/****** Configure kernel ******/
		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
										
		
		/****** Histogram Equalization ******/
		chrGPU.start();

		ChronoGPU chrCPU2;

		chrCPU2.start();
#ifdef OPTIM_TEXTURE_1D
		rgbToHsvCUDA<<< nbBlocks, nbThreads >>>(imgWidth, imgHeight, dev_hue, dev_saturation, dev_value);
#else
		rgbToHsvCUDA<<< nbBlocks, nbThreads >>>(dev_input, imgWidth, imgHeight, dev_hue, dev_saturation, dev_value);
#endif
		chrCPU2.stop();
		std::cout 	<< " RGB TO HSV Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

		chrCPU2.start();
#ifdef OPTIM_SHARED_MEM
		histogramCUDA<<< nbBlocks, nbThreads, bytesLevel >>>(dev_histogram, dev_value, imgWidth, imgHeight);
#else
		histogramCUDA<<< nbBlocks, nbThreads >>>(dev_histogram, dev_value, imgWidth, imgHeight);
#endif
		chrCPU2.stop();
		std::cout 	<< " HISTOGRAM Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

#ifdef OPTIM_CONSTANT_MEM
		int h_histogram[nbLevels];
		HANDLE_ERROR(cudaMemcpy(&h_histogram, dev_histogram, bytesLevel, cudaMemcpyDeviceToHost)); 
		HANDLE_ERROR(cudaMemcpyToSymbol(c_histogram, &h_histogram, bytesLevel));
#endif

		chrCPU2.start();
#ifdef OPTIM_CONSTANT_MEM
		cumulativeDistributionCUDA<<< 1, nbLevels, bytesLevel >>>(dev_repartition);
#else 
		cumulativeDistributionCUDA<<< 1, nbLevels, bytesLevel >>>(dev_histogram, dev_repartition);
#endif
		chrCPU2.stop();
		std::cout 	<< " REPARTITION Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

		chrCPU2.start();
		equalizationCUDA<<< nbBlocks, nbThreads >>>(dev_value, dev_repartition, imgWidth, imgHeight);
		chrCPU2.stop();
		std::cout 	<< " EQUALIZATION Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

		chrCPU2.start();
		hsvToRgbCUDA<<< nbBlocks, nbThreads >>>(dev_hue, dev_saturation, dev_value, imgWidth, imgHeight, dev_output);
		chrCPU2.stop();
		std::cout 	<< " HSV TO RGB Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;


		chrGPU.stop();
		std::cout   << "-> Done: " << chrGPU.elapsedTime() << " ms" << std::endl;



		/****** Check result ******/
		std::cout << "Checking result..." << std::endl;
		// Copy data from device to host (output array)   
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost)); 
		// Reset dev_output
		HANDLE_ERROR(cudaMemset(dev_output, 0, bytesImg));
		compareImages(resultCPU, output);


		/****** Free arrays on device ******/
		cudaFree(dev_input);
		cudaFree(dev_output);
		cudaFree(dev_hue);
		cudaFree(dev_saturation);
		cudaFree(dev_value);
		cudaFree(dev_histogram);
		cudaFree(dev_repartition);

#ifdef OPTIM_TEXTURE_1D
		HANDLE_ERROR(cudaUnbindTexture(t_in1D));
#endif
	}
}
