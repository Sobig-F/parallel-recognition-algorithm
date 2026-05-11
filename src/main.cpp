#include <filesystem>
#include <iostream>
#include <clocale>
#include <string>

#define NOMINMAX
#include <Windows.h>

#include "LBPH/lbph_descriptors.hpp"
#include "app/win_gui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "preproc/cropFace.hpp"

namespace
{
void ConfigureConsoleUtf8()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    std::setlocale(LC_CTYPE, ".UTF-8");
}

int SmokeTest()
{
    cv::Mat image = cv::imread(SRC_DIR "../data/Faces/Akshay Kumar_24.jpg", cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Smoke test image was not found." << std::endl;
        return 1;
    }

    preprocessing::FaceDetector detector(ONNX_MODEL_PATH);
    cv::Mat crop = detector.Detect(image);
    if (crop.empty())
    {
        std::cerr << "Smoke test face was not detected." << std::endl;
        return 1;
    }

    cv::Mat cpuPrepared = biometrics::PreprocessCpu(crop);
    cv::Mat cudaPrepared;
    if (biometrics::IsCudaAvailable())
    {
        cudaPrepared = biometrics::PreprocessCuda(crop);
    }

    for (const biometrics::AlgorithmInfo& algorithm : biometrics::Algorithms())
    {
        if (algorithm.usesCuda && cudaPrepared.empty())
        {
            std::cout << algorithm.title << ": skipped, CUDA unavailable" << std::endl;
            continue;
        }

        cv::Mat prepared = algorithm.usesCuda ? cudaPrepared : cpuPrepared;
        cv::Mat descriptor = biometrics::ComputeDescriptor(algorithm.id, prepared);
        double selfSimilarity = biometrics::CompareDescriptors(algorithm.id, descriptor, descriptor);
        std::cout << algorithm.title
                  << ": descriptor=" << descriptor.cols
                  << ", self_similarity=" << selfSimilarity
                  << std::endl;
    }

    return 0;
}
} // namespace

int main(int argc, char* argv[])
{
    try
    {
        ConfigureConsoleUtf8();

        if (argc > 1 && std::string(argv[1]) == "--smoke-test")
        {
            return SmokeTest();
        }

        return biometrics::RunGuiApp(
            std::filesystem::path(ONNX_MODEL_PATH),
            std::filesystem::path(SRC_DIR) / ".." / "data" / "enrolled_faces"
        );
    }
    catch (const std::exception& error)
    {
        std::cerr << "Application failed: " << error.what() << std::endl;
        return 1;
    }
}
