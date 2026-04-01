#ifndef PREPROC_HPP
#define PREPROC_HPP

#include "opencv2/opencv.hpp"

namespace preprocessing
{
#define TARGET_X 256
#define TARGET_Y 256

void GrayScale(cv::Mat& input_);
void GaussainBlur(cv::Mat& input_);
void CLAHE(cv::Mat& input_);
void Resize(cv::Mat& input_);
}

#endif