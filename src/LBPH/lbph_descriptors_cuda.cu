#include "LBPH/lbph_descriptors.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>

#include "cuda_runtime.h"
#include "opencv2/core.hpp"

namespace
{
constexpr int kGridX = 8;
constexpr int kGridY = 8;
constexpr int kNeighbors = 8;
constexpr int kOpenCvBins = 1 << kNeighbors;
constexpr int kCustomBins = 10;
constexpr int kCustomRadiiCount = 3;
constexpr int kCustomRadii[kCustomRadiiCount] = {1, 2, 3};
constexpr float kPi = 3.14159265358979323846f;

void CheckCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        std::ostringstream message;
        message << operation << " failed: " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

__device__ float ReadBilinear(const unsigned char* image, int rows, int cols, float y, float x)
{
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = min(x0 + 1, cols - 1);
    int y1 = min(y0 + 1, rows - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float dx = x - static_cast<float>(x0);
    float dy = y - static_cast<float>(y0);

    float topLeft = static_cast<float>(image[y0 * cols + x0]);
    float topRight = static_cast<float>(image[y0 * cols + x1]);
    float bottomLeft = static_cast<float>(image[y1 * cols + x0]);
    float bottomRight = static_cast<float>(image[y1 * cols + x1]);

    return topLeft * (1.0f - dx) * (1.0f - dy) +
           topRight * dx * (1.0f - dy) +
           bottomLeft * (1.0f - dx) * dy +
           bottomRight * dx * dy;
}

__device__ unsigned int LbpCode(const unsigned char* image, int rows, int cols, int row, int col, int radius)
{
    unsigned char center = image[row * cols + col];
    unsigned int code = 0;

    for (int p = 0; p < kNeighbors; ++p)
    {
        float angle = -static_cast<float>(p) * kPi / 4.0f;
        float x = static_cast<float>(col) + static_cast<float>(radius) * cosf(angle);
        float y = static_cast<float>(row) + static_cast<float>(radius) * sinf(angle);
        if (ReadBilinear(image, rows, cols, y, x) >= static_cast<float>(center))
        {
            code |= (1u << p);
        }
    }

    return code;
}

__device__ int CountTransitions(unsigned int code)
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

__device__ int UniformBin(unsigned int code)
{
    return CountTransitions(code) <= 2 ? __popc(code) : 9;
}

__global__ void BuildCustomHistogramKernel(
    const unsigned char* image,
    int rows,
    int cols,
    int radius,
    int radiusIndex,
    int outRows,
    int outCols,
    int cellH,
    int cellW,
    unsigned int* histogram
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outRows * outCols;
    if (idx >= total)
    {
        return;
    }

    int outY = idx / outCols;
    int outX = idx % outCols;
    int cellX = outX / cellW;
    int cellY = outY / cellH;
    if (cellX >= kGridX || cellY >= kGridY)
    {
        return;
    }

    int row = outY + radius;
    int col = outX + radius;
    int bin = UniformBin(LbpCode(image, rows, cols, row, col, radius));
    int perRadiusSize = kGridX * kGridY * kCustomBins;
    int histIndex = radiusIndex * perRadiusSize + (cellY * kGridX + cellX) * kCustomBins + bin;
    atomicAdd(&histogram[histIndex], 1u);
}

__global__ void BuildOpenCvStyleHistogramKernel(
    const unsigned char* image,
    int rows,
    int cols,
    int outRows,
    int outCols,
    int cellH,
    int cellW,
    unsigned int* histogram
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outRows * outCols;
    if (idx >= total)
    {
        return;
    }

    int outY = idx / outCols;
    int outX = idx % outCols;
    int cellX = outX / cellW;
    int cellY = outY / cellH;
    if (cellX >= kGridX || cellY >= kGridY)
    {
        return;
    }

    int row = outY + 1;
    int col = outX + 1;
    unsigned int bin = LbpCode(image, rows, cols, row, col, 1);
    int histIndex = (cellY * kGridX + cellX) * kOpenCvBins + static_cast<int>(bin);
    atomicAdd(&histogram[histIndex], 1u);
}

void UploadImageAndAllocateHistogram(
    const cv::Mat& gray,
    int histSize,
    unsigned char** deviceImage,
    unsigned int** deviceHistogram
)
{
    const std::size_t imageBytes = static_cast<std::size_t>(gray.rows) * gray.cols * sizeof(unsigned char);
    const std::size_t histogramBytes = static_cast<std::size_t>(histSize) * sizeof(unsigned int);

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(deviceImage), imageBytes), "cudaMalloc(image)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(deviceHistogram), histogramBytes), "cudaMalloc(histogram)");
    CheckCuda(cudaMemcpy(*deviceImage, gray.ptr<unsigned char>(), imageBytes, cudaMemcpyHostToDevice), "cudaMemcpy(image)");
    CheckCuda(cudaMemset(*deviceHistogram, 0, histogramBytes), "cudaMemset(histogram)");
}
} // namespace

namespace biometrics
{
cv::Mat ComputeCustomCudaDescriptor(const cv::Mat& gray)
{
    if (!IsCudaAvailable())
    {
        throw std::runtime_error("CUDA device is not available");
    }
    if (gray.empty() || gray.type() != CV_8UC1 || !gray.isContinuous())
    {
        throw std::invalid_argument("CUDA custom LBPH expects a continuous CV_8UC1 image");
    }

    constexpr int perRadiusSize = kGridX * kGridY * kCustomBins;
    constexpr int histSize = kCustomRadiiCount * perRadiusSize;
    unsigned char* deviceImage = nullptr;
    unsigned int* deviceHistogram = nullptr;

    try
    {
        UploadImageAndAllocateHistogram(gray, histSize, &deviceImage, &deviceHistogram);

        constexpr int threads = 256;
        for (int radiusIndex = 0; radiusIndex < kCustomRadiiCount; ++radiusIndex)
        {
            int radius = kCustomRadii[radiusIndex];
            int outRows = gray.rows - 2 * radius;
            int outCols = gray.cols - 2 * radius;
            int cellH = outRows / kGridY;
            int cellW = outCols / kGridX;
            if (cellH <= 0 || cellW <= 0)
            {
                throw std::invalid_argument("Image is too small for custom CUDA LBPH grid");
            }

            int total = outRows * outCols;
            int blocks = (total + threads - 1) / threads;
            BuildCustomHistogramKernel<<<blocks, threads>>>(
                deviceImage,
                gray.rows,
                gray.cols,
                radius,
                radiusIndex,
                outRows,
                outCols,
                cellH,
                cellW,
                deviceHistogram
            );
            CheckCuda(cudaGetLastError(), "BuildCustomHistogramKernel");
        }

        CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(custom)");

        std::vector<unsigned int> hostHistogram(histSize);
        CheckCuda(
            cudaMemcpy(hostHistogram.data(), deviceHistogram, histSize * sizeof(unsigned int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(custom histogram)"
        );

        cv::Mat descriptor = cv::Mat::zeros(1, histSize, CV_32FC1);
        float* descriptorData = descriptor.ptr<float>();
        for (int radiusIndex = 0; radiusIndex < kCustomRadiiCount; ++radiusIndex)
        {
            int base = radiusIndex * perRadiusSize;
            float sum = 0.0f;
            for (int i = 0; i < perRadiusSize; ++i)
            {
                sum += static_cast<float>(hostHistogram[base + i]);
            }
            if (sum <= 0.0f)
            {
                continue;
            }
            for (int i = 0; i < perRadiusSize; ++i)
            {
                descriptorData[base + i] = static_cast<float>(hostHistogram[base + i]) / sum;
            }
        }

        CheckCuda(cudaFree(deviceImage), "cudaFree(image)");
        CheckCuda(cudaFree(deviceHistogram), "cudaFree(histogram)");
        return descriptor;
    }
    catch (...)
    {
        cudaFree(deviceImage);
        cudaFree(deviceHistogram);
        throw;
    }
}

cv::Mat ComputeOpenCvCudaDescriptor(const cv::Mat& gray)
{
    if (!IsCudaAvailable())
    {
        throw std::runtime_error("CUDA device is not available");
    }
    if (gray.empty() || gray.type() != CV_8UC1 || !gray.isContinuous())
    {
        throw std::invalid_argument("CUDA OpenCV-style LBPH expects a continuous CV_8UC1 image");
    }

    constexpr int radius = 1;
    constexpr int histSize = kGridX * kGridY * kOpenCvBins;
    int outRows = gray.rows - 2 * radius;
    int outCols = gray.cols - 2 * radius;
    int cellH = outRows / kGridY;
    int cellW = outCols / kGridX;
    if (cellH <= 0 || cellW <= 0)
    {
        throw std::invalid_argument("Image is too small for OpenCV-style CUDA LBPH grid");
    }

    unsigned char* deviceImage = nullptr;
    unsigned int* deviceHistogram = nullptr;

    try
    {
        UploadImageAndAllocateHistogram(gray, histSize, &deviceImage, &deviceHistogram);

        constexpr int threads = 256;
        int total = outRows * outCols;
        int blocks = (total + threads - 1) / threads;
        BuildOpenCvStyleHistogramKernel<<<blocks, threads>>>(
            deviceImage,
            gray.rows,
            gray.cols,
            outRows,
            outCols,
            cellH,
            cellW,
            deviceHistogram
        );
        CheckCuda(cudaGetLastError(), "BuildOpenCvStyleHistogramKernel");
        CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(opencv-style)");

        std::vector<unsigned int> hostHistogram(histSize);
        CheckCuda(
            cudaMemcpy(hostHistogram.data(), deviceHistogram, histSize * sizeof(unsigned int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(opencv-style histogram)"
        );

        cv::Mat descriptor = cv::Mat::zeros(1, histSize, CV_32FC1);
        float* descriptorData = descriptor.ptr<float>();
        for (int cell = 0; cell < kGridX * kGridY; ++cell)
        {
            int base = cell * kOpenCvBins;
            float sum = 0.0f;
            for (int bin = 0; bin < kOpenCvBins; ++bin)
            {
                sum += static_cast<float>(hostHistogram[base + bin]);
            }
            if (sum <= 0.0f)
            {
                continue;
            }
            for (int bin = 0; bin < kOpenCvBins; ++bin)
            {
                descriptorData[base + bin] = static_cast<float>(hostHistogram[base + bin]) / sum;
            }
        }

        CheckCuda(cudaFree(deviceImage), "cudaFree(image)");
        CheckCuda(cudaFree(deviceHistogram), "cudaFree(histogram)");
        return descriptor;
    }
    catch (...)
    {
        cudaFree(deviceImage);
        cudaFree(deviceHistogram);
        throw;
    }
}
} // namespace biometrics
