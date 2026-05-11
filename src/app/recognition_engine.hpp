#ifndef RECOGNITION_ENGINE_HPP
#define RECOGNITION_ENGINE_HPP

#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "app/face_database.hpp"
#include "preproc/cropFace.hpp"

namespace biometrics
{
using ProgressCallback = std::function<void(const std::string&)>;

struct RecognitionThresholds
{
    double custom = 0.95;
    double openCv = 0.03;
};

struct AlgorithmRunSummary
{
    AlgorithmId algorithm;
    double milliseconds = 0.0;
    double averageMilliseconds = 0.0;
    double bestSimilarity = 0.0;
    int processed = 0;
    int failed = 0;
    int expectedPass = 0;
    int expectedDeny = 0;
    int passed = 0;
    int denied = 0;
    int correctPasses = 0;
    int correct = 0;
    int errors = 0;
    int falseAccepts = 0;
    int falseRejects = 0;
    int wrongMatches = 0;
    int correctDenies = 0;
};

struct ImageAlgorithmResult
{
    std::filesystem::path imagePath;
    std::string expectedName;
    bool expectedInDatabase = false;
    AlgorithmId algorithm;
    MatchResult match;
    bool recognized = false;
    bool correct = false;
    std::string outcome;
    std::string errorType;
    double threshold = 0.0;
    double totalMilliseconds = 0.0;
    std::string error;
};

struct SearchReport
{
    std::vector<AlgorithmRunSummary> summaries;
    std::vector<ImageAlgorithmResult> results;
};

class RecognitionEngine
{
public:
    RecognitionEngine(std::filesystem::path modelPath, std::filesystem::path databaseRoot);

    void ReloadDatabase();
    void EnrollFace(
        const std::string& name,
        const std::vector<std::filesystem::path>& imagePaths,
        const ProgressCallback& progress
    );
    SearchReport SearchDirectory(
        const std::filesystem::path& directory,
        const ProgressCallback& progress,
        const RecognitionThresholds& thresholds = {},
        const std::vector<AlgorithmId>& selectedAlgorithms = {}
    );
    void SaveReportCsv(const SearchReport& report, const std::filesystem::path& path) const;

    int KnownFacesCount() const;
    const FaceDatabase& Database() const noexcept;

private:
    struct CandidateFace
    {
        std::filesystem::path path;
        std::string expectedName;
        bool expectedInDatabase = false;
        cv::Mat crop;
    };

    cv::Mat LoadAndCropFace(const std::filesystem::path& path);
    std::vector<CandidateFace> LoadCandidates(const std::filesystem::path& directory, const ProgressCallback& progress);
    AlgorithmRunSummary RunAlgorithm(
        AlgorithmId algorithm,
        const std::vector<CandidateFace>& candidates,
        std::vector<ImageAlgorithmResult>& results,
        const ProgressCallback& progress,
        const RecognitionThresholds& thresholds
    ) const;

    preprocessing::FaceDetector _detector;
    FaceDatabase _database;
};
} // namespace biometrics

#endif
