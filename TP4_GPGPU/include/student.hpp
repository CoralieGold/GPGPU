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
	void studentJob(const std::vector<uchar3> &input, const uint imgWidth, const uint imgHeight, const uint nbLevels,
					const std::vector<uchar3> &resultCPU, std::vector<uchar3> &output);

	std::ostream &operator <<(std::ostream &os, const uchar3 &c);

	void compareImages(const std::vector<uchar3> &a, const std::vector<uchar3> &b);
}

#endif
