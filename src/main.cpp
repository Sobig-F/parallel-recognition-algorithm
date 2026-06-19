#include <filesystem>
#include <iostream>
#include <clocale>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

#define NOMINMAX
#include <Windows.h>

#include "LBPH/lbph_descriptors.hpp"
#include "app/recognition_engine.hpp"
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

int EngineTest()
{
    namespace fs = std::filesystem;

    fs::path sourceDir = fs::path(SRC_DIR) / ".." / "data" / "Faces";
    fs::path trainImage = sourceDir / "Akshay Kumar_24.jpg";
    fs::path probeImage = sourceDir / "Akshay Kumar_25.jpg";
    if (!fs::exists(trainImage) || !fs::exists(probeImage))
    {
        std::cerr << "Engine test images were not found." << std::endl;
        return 1;
    }

    fs::path testRoot = fs::temp_directory_path() / "biometric_engine_self_test";
    fs::remove_all(testRoot);
    fs::create_directories(testRoot / "db");
    fs::create_directories(testRoot / "probe");
    fs::copy_file(probeImage, testRoot / "probe" / probeImage.filename(), fs::copy_options::overwrite_existing);

    biometrics::RecognitionEngine engine(ONNX_MODEL_PATH, testRoot / "db");
    engine.EnrollFace("Akshay Kumar", {trainImage}, [](const std::string& message) {
        std::cout << message << std::endl;
    });

    biometrics::RecognitionThresholds thresholds;
    thresholds.custom = 0.90;
    thresholds.openCv = 0.02;

    biometrics::SearchReport report = engine.SearchDirectory(
        testRoot / "probe",
        [](const std::string& message) {
            std::cout << message << std::endl;
        },
        thresholds
    );

    bool ok = !report.summaries.empty();
    for (const biometrics::AlgorithmRunSummary& summary : report.summaries)
    {
        std::cout << biometrics::AlgorithmTitle(summary.algorithm)
                  << ": processed=" << summary.processed
                  << ", correct=" << summary.correct
                  << ", errors=" << summary.errors
                  << ", total_ms=" << summary.milliseconds
                  << std::endl;
        ok = ok && summary.processed == 1 && summary.correct == 1;
    }

    fs::remove_all(testRoot);
    return ok ? 0 : 1;
}

std::string TimestampForFile()
{
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_s(&local, &time);

    std::ostringstream out;
    out << std::put_time(&local, "%Y%m%d_%H%M%S");
    return out.str();
}

int BatchRun(int argc, char* argv[])
{
    namespace fs = std::filesystem;

    fs::path folder = argc > 2 ? fs::path(argv[2]) : fs::path(SRC_DIR) / ".." / "data" / "Faces";
    biometrics::RecognitionEngine engine(
        fs::path(ONNX_MODEL_PATH),
        fs::path(SRC_DIR) / ".." / "face_database"
    );

    biometrics::SearchReport report = engine.SearchDirectory(folder, {});
    fs::path csvPath = fs::path(SRC_DIR) / ".." / "result" / "research_reports" /
        ("face_control_cli_" + TimestampForFile() + ".csv");
    engine.SaveReportCsv(report, csvPath);

    std::cout << "CSV report: " << csvPath.string() << std::endl;
    for (const biometrics::AlgorithmRunSummary& summary : report.summaries)
    {
        std::cout << biometrics::AlgorithmTitle(summary.algorithm)
                  << ": total_ms=" << summary.milliseconds
                  << ", avg_ms=" << summary.averageMilliseconds
                  << ", processed=" << summary.processed
                  << ", correct=" << summary.correct
                  << ", errors=" << summary.errors
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
        if (argc > 1 && std::string(argv[1]) == "--engine-test")
        {
            return EngineTest();
        }
        if (argc > 1 && std::string(argv[1]) == "--batch")
        {
            return BatchRun(argc, argv);
        }

        return biometrics::RunGuiApp(
            std::filesystem::path(ONNX_MODEL_PATH),
            std::filesystem::path(SRC_DIR) / ".." / "face_database"
        );
    }
    catch (const std::exception& error)
    {
        std::cerr << "Application failed: " << error.what() << std::endl;
        return 1;
    }
}
