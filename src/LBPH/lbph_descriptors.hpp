#ifndef LBPH_DESCRIPTORS_HPP
#define LBPH_DESCRIPTORS_HPP

#include <array>
#include <string>
#include <vector>

#include "opencv2/core.hpp"

namespace biometrics
{
enum class AlgorithmId
{
    CustomCpu = 0,
    CustomCuda,
    OpenCvCpu,
    OpenCvCuda
};

struct AlgorithmInfo
{
    AlgorithmId id;
    const char* key;
    const char* title;
    bool usesCuda;
};

const std::array<AlgorithmInfo, 4>& Algorithms();
const AlgorithmInfo& GetAlgorithmInfo(AlgorithmId id);
std::string AlgorithmKey(AlgorithmId id);
std::string AlgorithmTitle(AlgorithmId id);

cv::Mat PreprocessCpu(const cv::Mat& input);
cv::Mat PreprocessCuda(const cv::Mat& input);

cv::Mat ComputeDescriptor(AlgorithmId id, const cv::Mat& preparedGray);
double CompareDescriptors(AlgorithmId id, const cv::Mat& first, const cv::Mat& second);
bool IsCudaAvailable();
} // namespace biometrics

#endif
