#include "LBPH/lbph_descriptors.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cfloat>
#include <numbers>
#include <numeric>
#include <stdexcept>

#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"

namespace biometrics
{
cv::Mat ComputeCustomCudaDescriptor(const cv::Mat& gray);
cv::Mat ComputeOpenCvCudaDescriptor(const cv::Mat& gray);

namespace
{
constexpr int kTargetWidth = 256;
constexpr int kTargetHeight = 256;
constexpr int kGridX = 8;
constexpr int kGridY = 8;
constexpr int kNeighbors = 8;
constexpr int kOpenCvBins = 1 << kNeighbors;
constexpr int kCustomBins = 10;
constexpr std::array<int, 3> kCustomRadii{1, 2, 3};

int CountTransitions(unsigned int code)
{
    int transitions = 0;
    for (int bit = 0; bit < kNeighbors; ++bit)
    {
        int current = (code >> bit) & 1;
        int next = (code >> ((bit + 1) % kNeighbors)) & 1;
        if (current != next)
        {
            ++transitions;
        }
    }

    return transitions;
}

int UniformBin(unsigned int code)
{
    if (CountTransitions(code) <= 2)
    {
        return std::popcount(code);
    }

    return 9;
}

float ReadBilinear(const cv::Mat& image, float y, float x)
{
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, image.cols - 1);
    int y1 = std::min(y0 + 1, image.rows - 1);
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);

    float dx = x - static_cast<float>(x0);
    float dy = y - static_cast<float>(y0);

    return static_cast<float>(image.at<uchar>(y0, x0)) * (1.0f - dx) * (1.0f - dy) +
           static_cast<float>(image.at<uchar>(y0, x1)) * dx * (1.0f - dy) +
           static_cast<float>(image.at<uchar>(y1, x0)) * (1.0f - dx) * dy +
           static_cast<float>(image.at<uchar>(y1, x1)) * dx * dy;
}

unsigned int LbpCode(const cv::Mat& image, int row, int col, int radius)
{
    static constexpr float angles[kNeighbors] = {
        0.0f,
        -std::numbers::pi_v<float> / 4.0f,
        -std::numbers::pi_v<float> / 2.0f,
        -3.0f * std::numbers::pi_v<float> / 4.0f,
        std::numbers::pi_v<float>,
        3.0f * std::numbers::pi_v<float> / 4.0f,
        std::numbers::pi_v<float> / 2.0f,
        std::numbers::pi_v<float> / 4.0f
    };

    uchar center = image.at<uchar>(row, col);
    unsigned int code = 0;
    for (int p = 0; p < kNeighbors; ++p)
    {
        float x = static_cast<float>(col) + static_cast<float>(radius) * std::cos(angles[p]);
        float y = static_cast<float>(row) + static_cast<float>(radius) * std::sin(angles[p]);
        if (ReadBilinear(image, y, x) >= static_cast<float>(center))
        {
            code |= (1u << p);
        }
    }

    return code;
}

cv::Mat EnsureGray8(const cv::Mat& input)
{
    if (input.empty())
    {
        throw std::invalid_argument("Input image is empty");
    }

    cv::Mat gray;
    if (input.channels() == 1)
    {
        gray = input;
    }
    else if (input.channels() == 3)
    {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else if (input.channels() == 4)
    {
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw std::invalid_argument("Unsupported image channel count");
    }

    cv::Mat result;
    if (gray.type() == CV_8UC1)
    {
        result = gray;
    }
    else
    {
        gray.convertTo(result, CV_8UC1);
    }

    return result.isContinuous() ? result : result.clone();
}

cv::Mat ComputeCustomCpuDescriptor(const cv::Mat& gray)
{
    const int cells = kGridX * kGridY;
    const int perRadiusSize = cells * kCustomBins;
    cv::Mat descriptor = cv::Mat::zeros(1, static_cast<int>(kCustomRadii.size()) * perRadiusSize, CV_32FC1);
    float* data = descriptor.ptr<float>();

    for (int radiusIndex = 0; radiusIndex < static_cast<int>(kCustomRadii.size()); ++radiusIndex)
    {
        int radius = kCustomRadii[radiusIndex];
        int validRows = gray.rows - 2 * radius;
        int validCols = gray.cols - 2 * radius;
        int cellH = validRows / kGridY;
        int cellW = validCols / kGridX;
        if (cellH <= 0 || cellW <= 0)
        {
            throw std::invalid_argument("Image is too small for custom LBPH grid");
        }

        int base = radiusIndex * perRadiusSize;
        for (int gridY = 0; gridY < kGridY; ++gridY)
        {
            for (int gridX = 0; gridX < kGridX; ++gridX)
            {
                int rowStart = radius + gridY * cellH;
                int colStart = radius + gridX * cellW;
                int cellBase = base + (gridY * kGridX + gridX) * kCustomBins;

                for (int row = rowStart; row < rowStart + cellH; ++row)
                {
                    for (int col = colStart; col < colStart + cellW; ++col)
                    {
                        int bin = UniformBin(LbpCode(gray, row, col, radius));
                        data[cellBase + bin] += 1.0f;
                    }
                }
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < perRadiusSize; ++i)
        {
            sum += data[base + i];
        }
        if (sum > 0.0f)
        {
            for (int i = 0; i < perRadiusSize; ++i)
            {
                data[base + i] /= sum;
            }
        }
    }

    return descriptor;
}

cv::Mat ComputeOpenCvCpuDescriptor(const cv::Mat& gray)
{
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer =
        cv::face::LBPHFaceRecognizer::create(1, kNeighbors, kGridX, kGridY, DBL_MAX);

    std::vector<cv::Mat> images{gray};
    cv::Mat labels = (cv::Mat_<int>(1, 1) << 0);
    recognizer->train(images, labels);

    std::vector<cv::Mat> histograms = recognizer->getHistograms();
    if (histograms.empty())
    {
        throw std::runtime_error("OpenCV LBPH did not return a histogram");
    }

    cv::Mat descriptor = histograms.front().reshape(1, 1).clone();
    if (descriptor.type() != CV_32FC1)
    {
        descriptor.convertTo(descriptor, CV_32FC1);
    }

    return descriptor;
}

double CompareOpenCvLike(const cv::Mat& first, const cv::Mat& second)
{
    double distance = cv::compareHist(first, second, cv::HISTCMP_CHISQR_ALT);
    return 1.0 / (1.0 + distance);
}

double CompareCustom(const cv::Mat& first, const cv::Mat& second)
{
    constexpr double weights[3] = {0.5, 0.3, 0.2};
    constexpr double alpha = 0.6;
    constexpr double beta = 1.0 - alpha;
    constexpr int perRadiusSize = kGridX * kGridY * kCustomBins;

    if (first.cols != second.cols || first.cols != perRadiusSize * static_cast<int>(kCustomRadii.size()))
    {
        throw std::invalid_argument("Custom LBPH descriptors have incompatible sizes");
    }

    const float* aData = first.ptr<float>();
    const float* bData = second.ptr<float>();
    double chi2Total = 0.0;
    double cosineTotal = 0.0;

    for (int radiusIndex = 0; radiusIndex < static_cast<int>(kCustomRadii.size()); ++radiusIndex)
    {
        int offset = radiusIndex * perRadiusSize;
        double chi2Sum = 0.0;
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < perRadiusSize; ++i)
        {
            double a = aData[offset + i];
            double b = bData[offset + i];
            dot += a * b;
            normA += a * a;
            normB += b * b;

            double sum = a + b;
            if (sum > 1e-10)
            {
                chi2Sum += (a - b) * (a - b) / sum;
            }
        }

        double denom = std::sqrt(normA) * std::sqrt(normB);
        double cosine = denom > 1e-10 ? dot / denom : 0.0;
        double chi2 = 1.0 / (1.0 + chi2Sum);

        chi2Total += weights[radiusIndex] * chi2;
        cosineTotal += weights[radiusIndex] * cosine;
    }

    return alpha * chi2Total + beta * cosineTotal;
}
} // namespace

const std::array<AlgorithmInfo, 4>& Algorithms()
{
    static constexpr std::array<AlgorithmInfo, 4> algorithms{{
        {AlgorithmId::CustomCpu, "custom_cpu", "Моя LBPH CPU", false},
        {AlgorithmId::CustomCuda, "custom_cuda", "Моя LBPH CUDA", true},
        {AlgorithmId::OpenCvCpu, "opencv_cpu", "OpenCV LBPH CPU", false},
        {AlgorithmId::OpenCvCuda, "opencv_cuda", "OpenCV LBPH CUDA", true}
    }};

    return algorithms;
}

const AlgorithmInfo& GetAlgorithmInfo(AlgorithmId id)
{
    const auto& algorithms = Algorithms();
    auto found = std::find_if(algorithms.begin(), algorithms.end(), [id](const AlgorithmInfo& info) {
        return info.id == id;
    });

    if (found == algorithms.end())
    {
        throw std::invalid_argument("Unknown algorithm id");
    }

    return *found;
}

std::string AlgorithmKey(AlgorithmId id)
{
    return GetAlgorithmInfo(id).key;
}

std::string AlgorithmTitle(AlgorithmId id)
{
    return GetAlgorithmInfo(id).title;
}

cv::Mat PreprocessCpu(const cv::Mat& input)
{
    cv::Mat gray = EnsureGray8(input);
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(kTargetWidth, kTargetHeight), 0.0, 0.0, cv::INTER_LINEAR);
    return resized.isContinuous() ? resized : resized.clone();
}

cv::Mat PreprocessCuda(const cv::Mat& input)
{
    if (!IsCudaAvailable())
    {
        throw std::runtime_error("CUDA device is not available");
    }

    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(input);

    cv::cuda::GpuMat gpuGray;
    if (input.channels() == 1)
    {
        gpuGray = gpuInput;
    }
    else if (input.channels() == 3)
    {
        cv::cuda::cvtColor(gpuInput, gpuGray, cv::COLOR_BGR2GRAY);
    }
    else if (input.channels() == 4)
    {
        cv::cuda::cvtColor(gpuInput, gpuGray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw std::invalid_argument("Unsupported image channel count");
    }

    cv::cuda::GpuMat gpuResized;
    cv::cuda::resize(gpuGray, gpuResized, cv::Size(kTargetWidth, kTargetHeight), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat result;
    gpuResized.download(result);
    return result.isContinuous() ? result : result.clone();
}

cv::Mat ComputeDescriptor(AlgorithmId id, const cv::Mat& preparedGray)
{
    cv::Mat gray = EnsureGray8(preparedGray);
    switch (id)
    {
    case AlgorithmId::CustomCpu:
        return ComputeCustomCpuDescriptor(gray);
    case AlgorithmId::OpenCvCpu:
        return ComputeOpenCvCpuDescriptor(gray);
    case AlgorithmId::CustomCuda:
        return ComputeCustomCudaDescriptor(gray);
    case AlgorithmId::OpenCvCuda:
        return ComputeOpenCvCudaDescriptor(gray);
    }

    throw std::invalid_argument("Unknown algorithm id");
}

double CompareDescriptors(AlgorithmId id, const cv::Mat& first, const cv::Mat& second)
{
    if (first.empty() || second.empty())
    {
        return 0.0;
    }

    cv::Mat first32 = first.reshape(1, 1);
    cv::Mat second32 = second.reshape(1, 1);
    if (first32.type() != CV_32FC1)
    {
        first32.convertTo(first32, CV_32FC1);
    }
    if (second32.type() != CV_32FC1)
    {
        second32.convertTo(second32, CV_32FC1);
    }
    if (first32.cols != second32.cols)
    {
        throw std::invalid_argument("LBPH descriptors have incompatible sizes");
    }

    switch (id)
    {
    case AlgorithmId::CustomCpu:
    case AlgorithmId::CustomCuda:
        return CompareCustom(first32, second32);
    case AlgorithmId::OpenCvCpu:
    case AlgorithmId::OpenCvCuda:
        return CompareOpenCvLike(first32, second32);
    }

    throw std::invalid_argument("Unknown algorithm id");
}

bool IsCudaAvailable()
{
    try
    {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    }
    catch (...)
    {
        return false;
    }
}
} // namespace biometrics
