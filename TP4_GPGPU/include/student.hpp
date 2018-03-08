/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
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
	// Init data andlaunches kernels
    void studentJob(const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
                    const std::vector<uchar4> &resCPU, // Just for comparison
                    std::vector<uchar4> &output);
}

#endif
