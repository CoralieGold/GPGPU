/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.hpp
* Author: Maxime MARIA
*/


#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"

namespace IMAC
{
	// Kernel:
	// - input: input image RGB
	// - output: output image RGB
    __global__
    void sepiaCUDA(const uchar *const input, const uint width, const uint height, uchar *const output );

	// - input: input image RGB
	// - output: output image RGB
    void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output);

}

#endif
