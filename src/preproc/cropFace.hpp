#ifndef CROP_FACE_HPP
#define CROP_FACE_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/filesystem.hpp"
// #include "opencv2/dnn.hpp"

namespace preprocessing
{
#define BLOBX 640
#define BLOBY 480

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
} // namespace preprocessing

#endif