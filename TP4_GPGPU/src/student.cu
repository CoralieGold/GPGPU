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
	// For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar3 &c) {
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "]";  
    	return os; 
	}

	__global__ void rgbToHsvCUDA(const uchar3 *const input, const uint imgWidth, const uint imgHeight, 
								float *hue, float *saturation, float *value) {

		
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const int idPixel = x + y * imgWidth;

				float red = (float)input[idPixel].x / 255.f;
				float green = (float)input[idPixel].y / 255.f;
				float blue = (float)input[idPixel].z / 255.f;

				float cMax = fmax(fmax(red, green), blue);
				float cMin = fmin(fmin(red, green), blue);
				float delta = cMax - cMin;

				hue[idPixel] = 0.f;
				if(cMax == red)   hue[idPixel] = 60.f * fmod(((green - blue) / delta), 6.f);
				else if(cMax == green) hue[idPixel] = 60.f * (((blue - red) / delta) + 2.f);
				else if(cMax == blue)  hue[idPixel] = 60.f * (((red - green) / delta) + 4.f);
				else hue[idPixel] = 0.f;
				if(hue[idPixel] < 0.f) hue[idPixel] += 360.f;

				if(cMax != 0.f) saturation[idPixel] = (float)(delta / cMax);
				else saturation[idPixel] = 0.f;

				value[idPixel] = cMax;
			}
		}
	}

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
					blue = 0;
				}
				else if(hue[idOut] >= 60.f && hue[idOut] < 120.f) {
					red = x;
					green = c;
					blue = 0;
				}
				else if(hue[idOut] >= 120.f && hue[idOut] < 180.f) {
					red = 0;
					green = c;
					blue = x;
				}
				else if(hue[idOut] >= 180.f && hue[idOut] < 240.f) {
					red = 0;
					green = x;
					blue = c;
				}
				else if(hue[idOut] >= 240.f && hue[idOut] < 300.f) {
					red = x;
					green = 0;
					blue = c;
				}
				else {
					red = c;
					green = 0;
					blue = x;
				}

				output[idOut].x = (uchar)((red + m) * 255.f);
				output[idOut].y = (uchar)((green + m) * 255.f);
				output[idOut].z = (uchar)((blue + m) * 255.f);
			}
		}
	}

	__global__ void cumulativeDistributionCUDA(const int *const histogram, int *repartition, const uint nbLevels) {
		int distribution = 0;
		for(int level = 0; level < nbLevels; ++ level) {
			distribution += histogram[level];
			repartition[level] = distribution;
		}
	}

	__global__ void histogramCUDA(int *histogram, const float *const value, const uint size) {
	    for(int pixel = 0; pixel < size; pixel++) {
	 		histogram[(int)value[pixel]*255] += 1;     
	    }
	}

	__global__ void equalizationCUDA(float *value, const int *const repartition, const uint size) {
		for(int pixel = 0; pixel < size; pixel ++) {
			value[pixel] = (repartition[(int)value[pixel]*255] - repartition[0])/((float)size-1);
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
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].z - b[i].z) > 2 )
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
		std::cout 	<< "Allocating input and output images on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));

		const size_t bytesFloatImg = imageSize * sizeof(float);
		std::cout 	<< "Allocating hue, saturation and value on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_hue, bytesFloatImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_saturation, bytesFloatImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_value, bytesFloatImg));

		const size_t bytesLevel = nbLevels * sizeof(int);
		std::cout 	<< "Allocating histogram and repartition on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_histogram, bytesLevel));
		HANDLE_ERROR(cudaMalloc((void**)&dev_repartition, bytesLevel));

		std::cout << "Copy data from host to device" << std::endl;
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), bytesImg, cudaMemcpyHostToDevice));


		/****** Configure kernel ******/
		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
										
		
		/****** Histogram Equalization ******/
		chrGPU.start();

		rgbToHsvCUDA<<< nbBlocks, nbThreads >>>(dev_input, imgWidth, imgHeight, dev_hue, dev_saturation, dev_value);

		// histogramCUDA<<< nbBlocks, nbThreads >>>(dev_histogram, dev_value, imageSize);

		// cumulativeDistributionCUDA<<< nbBlocks, nbThreads >>>(dev_histogram, dev_repartition, nbLevels);

		// equalizationCUDA<<< nbBlocks, nbThreads >>>(dev_value, dev_repartition, imageSize);

		hsvToRgbCUDA<<< nbBlocks, nbThreads >>>(dev_hue, dev_saturation, dev_value, imgWidth, imgHeight, dev_output);

		chrGPU.stop();
		std::cout 	<< "-> Done: " << chrGPU.elapsedTime() << " ms" << std::endl;



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
	}
}
