#ifndef CROP_FACE_HPP
#define CROP_FACE_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/filesystem.hpp"
// #include "opencv2/dnn.hpp"

namespace preprocessing
{
#define BLOBX 640
#define BLOBY 480
#define TARGET_X 128
#define TARGET_Y 128

class FaceDetector
{
public:
    explicit FaceDetector(const std::string& modelPath_);
    cv::Mat Detect(const cv::Mat& input_);
    
private:
    cv::Ptr<cv::FaceDetectorYN> _net;
    float _confidenceThreshold = 0.6;
    bool _init = false;
};

void GrayScale(cv::Mat& input_);
void GaussainBlur(cv::Mat& input_);
void CLAHE(cv::Mat& input_);
void Resize(cv::Mat& input_);

} // namespace preprocessing

#endif