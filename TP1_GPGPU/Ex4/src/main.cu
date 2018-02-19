/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: main.cpp
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>

#include "student.hpp"
#include "chronoCPU.hpp"

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t <W> <H>: <W> width and <H> height of matrices" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

	// Computes matrices addition of 'input' and stores result in 'output'
	void matricesAddCPU(int** input1, int** input2, const uint width, const uint height, int** output) {
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();

		for(int i = 0; i < width; ++i) {
			for(int j = 0; j < height; ++j) {
				output[i][j] = input1[i][j] + input2[i][j];
			}
		}


		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// Compare two vectors
	bool compare(int** &a, int** &b, uint width, uint height)
	{
		for (uint i = 0; i < width; ++i)
		{
			for (uint j = 0; j < height; ++j)
			{
				// Floating precision can cause small difference between host and device
				if (std::abs(a[i][j] - b[i][j]) > 1)
				{
					std::cout << "Error at index " << i << ": a = " << a[i][j] << " - b = " << b[i][j] << std::endl;
					return false; 
				}
			}
		}
		return true;
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		int height;
		int width;

		// Parse command line
		if (argc < 3) 
		{
			std::cerr << "Please give a width AND a height" << std::endl;
			printUsageAndExit(argv[0]);
		}

		width = atoi(argv[1]);
		height = atoi(argv[2]);
		
		// Get input image
		int** input1 = new int*[width];
		int** input2 = new int*[width];

		// Create 2 output images
		int** outputGPU = new int*[width];
		int** outputCPU = new int*[width];

		// Init two input matrices
		input1[0] = new int[width * height];
		input2[0] = new int[width * height];
		outputGPU[0] = new int[width * height];
		outputCPU[0] = new int[width * height];

		for(int i = 1; i < height; ++i){
			input1[i] = input1[i-1] + width;
			input2[i] = input2[i-1] + width;
			outputGPU[i] = outputGPU[i-1] + width;
			outputCPU[i] = outputCPU[i-1] + width;
		}

		for(int i = 0; i < width; ++i) {
			for(int j = 0; j < height; ++j) {
				input1[i][j] = rand() % 100;
				input2[i][j] = rand() % 100;
			}
		}

		// Computation on CPU
		matricesAddCPU(input1, input2, width, height, outputCPU);

		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		studentJob(input1[0], input2[0], width, height, outputGPU[0]);
		
		std::cout << "============================================"	<< std::endl << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compare(outputCPU, outputGPU, width, height))
		{
			std::cout << " -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
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
