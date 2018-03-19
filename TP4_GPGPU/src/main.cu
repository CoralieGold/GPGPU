/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: main.cu
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>
#include <math.h>

#include "student.hpp"
#include "chronoCPU.hpp"
#include "lodepng.h"
#include "conv_utils.hpp"

namespace IMAC
{
	int NB_LEVELS = 256;

	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -f <F>: <F> image file name (required)" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Compate two arrays (a and b) of size n. Return true if equal
	bool compare(std::vector<float> &a, std::vector<float> &b, const int n)
	{
		for (int i = 0; i < n; ++i)
		{
			if (a[i] != b[i])
			{
				std::cout << "Error at index " << i  << ": a = " << a[i] << " - b = " << b[i] << std::endl;
				return false; 
			}
		}
		return true;
	}


	void rgbToHsvCPU(const std::vector<uchar3> &input, std::vector<float> &hue, std::vector<float> &saturation, std::vector<float> &value) {
		float red, green, blue, cMax, cMin, delta;
		for(int pixel = 0; pixel < input.size(); pixel ++) {

			red = (float)input[pixel].x / 255.f;
			green = (float)input[pixel].y / 255.f;
			blue = (float)input[pixel].z / 255.f;

			cMax = (float)std::max(std::max(red, green), blue);
			cMin = (float)std::min(std::min(red, green), blue);
			delta = (float)(cMax - cMin);

			hue[pixel] = 0.f;
			if(cMax == red)   hue[pixel] = 60.f * fmodf(((green - blue) / delta), 6.f);
			else if(cMax == green) hue[pixel] = 60.f * (((blue - red) / delta) + 2.f);
			else if(cMax == blue)  hue[pixel] = 60.f * (((red - green) / delta) + 4.f);
			else hue[pixel] = 0.f;
			if(hue[pixel] < 0) hue[pixel] += 360.f;

			if(cMax != 0.f) saturation[pixel] = (float)(delta / cMax);
			else saturation[pixel] = 0.f;

			value[pixel] = cMax;
		}
	}

	void hsvToRgbCPU(const std::vector<float> &hue, const std::vector<float> &saturation, const std::vector<float> &value, std::vector<uchar3> &output) {
		float c, x, m, red, blue, green;
		for(int pixel = 0; pixel < output.size(); pixel ++) {
			c = (float)(value[pixel] * saturation[pixel]);

			x = c * (1.f - fabs(fmodf((float)hue[pixel] / 60.f, 2.f) - 1.f));
			m = (float)(value[pixel] - c);

			if(hue[pixel] < 60.f) {
				red = c;
				green = x;
				blue = 0;
			}
			else if(hue[pixel] >= 60.f && hue[pixel] < 120.f) {
				red = x;
				green = c;
				blue = 0;
			}
			else if(hue[pixel] >= 120.f && hue[pixel] < 180.f) {
				red = 0;
				green = c;
				blue = x;
			}
			else if(hue[pixel] >= 180.f && hue[pixel] < 240.f) {
				red = 0;
				green = x;
				blue = c;
			}
			else if(hue[pixel] >= 240.f && hue[pixel] < 300.f) {
				red = x;
				green = 0;
				blue = c;
			}
			else {
				red = c;
				green = 0;
				blue = x;
			}

			output[pixel].x = (uchar)((red + m) * 255.f);
			output[pixel].y = (uchar)((green + m) * 255.f);
			output[pixel].z = (uchar)((blue + m) * 255.f);
		}
	}

	void cumulativeDistributionCPU(const std::vector<int> &histogram, std::vector<int> &repartition) {
		int distribution = 0;
		for(int level = 0; level < NB_LEVELS; ++ level) {
			distribution += histogram[level];
			repartition[level] = distribution;
		}
	}

	void histogramCPU(std::vector<int> &histogram, const std::vector<float> &value) {
	    for(int pixel = 0; pixel < value.size(); pixel++) {
	 		histogram[value[pixel]*255] += 1;     
	    }
	}

	void equalizationCPU(std::vector<float> &value, const std::vector<int> &repartition) {
		for(int pixel = 0; pixel < value.size(); pixel ++) {
			value[pixel] = (repartition[value[pixel]*255] - repartition[0])/((float)value.size()-1);
		}
	}

	void histogramEqualizationCPU(const std::vector<uchar3> &input, const uint imgWidth, const uint imgHeight, std::vector<uchar3> &output)
	{
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		
		int size = imgWidth*imgHeight;

		std::vector<float> hue(size);
		std::vector<float> saturation(size);
		std::vector<float> value(size);

		ChronoCPU chrCPU2;
		chrCPU2.start();
		rgbToHsvCPU(input, hue, saturation, value);
		chrCPU2.stop();
		std::cout 	<< " RGB TO HSV Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

		std::vector<int> histogram(NB_LEVELS);
		chrCPU2.start();
		histogramCPU(histogram, value);
		chrCPU2.stop();
		std::cout 	<< " HISTOGRAM Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

		std::vector<int> repartition(NB_LEVELS);
		chrCPU2.start();
		cumulativeDistributionCPU(histogram, repartition);
		chrCPU2.stop();
		std::cout 	<< " REPARTITION Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

		chrCPU2.start();
		equalizationCPU(value, repartition);
		chrCPU2.stop();
		std::cout 	<< " EQUALIZATION Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

		// std::vector<int> histogramEqualized(NB_LEVELS);
		// histogramCPU(histogramEqualized, value);

		chrCPU2.start();
		hsvToRgbCPU(hue, saturation, value, output);
		chrCPU2.stop();
		std::cout 	<< " HSV TO RGB Done : " << chrCPU2.elapsedTime() << " ms" << std::endl << std::endl;

		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		char fileName[2048];

		// Parse command line
		if (argc == 1) 
		{
			std::cerr << "Please give a file..." << std::endl;
			printUsageAndExit(argv[0]);
		}

		for (int i = 1; i < argc; ++i) 
		{
			if (!strcmp(argv[i], "-f")) 
			{
				if (sscanf(argv[++i], "%s", fileName) != 1)
				{
					printUsageAndExit(argv[0]);
				}
			}
			else
			{
				printUsageAndExit(argv[0]);
			}
		}
		
		// Get input image
		std::vector<uchar> inputUchar;
		uint imgWidth;
		uint imgHeight;

		std::cout << "Loading " << fileName << std::endl;
		unsigned error = lodepng::decode(inputUchar, imgWidth, imgHeight, fileName, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		// Convert to uchar3 for exercise convenience
		std::vector<uchar3> input;
		input.resize(inputUchar.size() / 3);
		for (uint i = 0; i < input.size(); ++i)
		{
			const uint id = 3 * i;
			input[i].x = inputUchar[id];
			input[i].y = inputUchar[id + 1];
			input[i].z = inputUchar[id + 2];
		}
		inputUchar.clear();
		std::cout << "Image has " << imgWidth << " x " << imgHeight << " pixels (RGB)" << std::endl;

		// Create 2 output images
		std::vector<uchar3> outputCPU(imgWidth * imgHeight);
		std::vector<uchar3> outputGPU(imgWidth * imgHeight);

		
		std::cout << input.size() << " - " << outputCPU.size() << " - " << outputGPU.size() << std::endl;

		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string outputCPUName = name + "_CPU" + ext;
		std::string outputGPUName = name + "_GPU" + ext;

		// Computation on CPU
		histogramEqualizationCPU(input, imgWidth, imgHeight, outputCPU);
	
		std::cout << "Save image as: " << outputCPUName << std::endl;
		error = lodepng::encode(outputCPUName, reinterpret_cast<uchar *>(outputCPU.data()), imgWidth, imgHeight, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::encode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		studentJob(input, imgWidth, imgHeight, NB_LEVELS, outputCPU, outputGPU);

		std::cout << "Save image as: " << outputGPUName << std::endl;
		error = lodepng::encode(outputGPUName, reinterpret_cast<uchar *>(outputGPU.data()), imgWidth, imgHeight, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout << "============================================"	<< std::endl << std::endl;
	}
}

int main(int argc, char **argv) 
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
