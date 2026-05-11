#include "app/recognition_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <future>
#include <iomanip>
#include <locale>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "opencv2/imgcodecs.hpp"

namespace biometrics
{
namespace
{
bool IsImageFile(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    for (char& ch : extension)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    return extension == ".jpg" ||
           extension == ".jpeg" ||
           extension == ".png" ||
           extension == ".bmp" ||
           extension == ".webp" ||
           extension == ".tif" ||
           extension == ".tiff";
}

std::string FormatDouble(double value, int precision = 4)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

double ClassRecallPercent(int correct, int total)
{
    return total > 0 ? 100.0 * static_cast<double>(correct) / static_cast<double>(total) : 0.0;
}

double BalancedAccuracyPercent(const AlgorithmRunSummary& summary)
{
    const bool hasPassClass = summary.expectedPass > 0;
    const bool hasDenyClass = summary.expectedDeny > 0;
    const double passRecall = ClassRecallPercent(summary.correctPasses, summary.expectedPass);
    const double denyRecall = ClassRecallPercent(summary.correctDenies, summary.expectedDeny);

    if (hasPassClass && hasDenyClass)
    {
        return (passRecall + denyRecall) / 2.0;
    }
    if (hasPassClass)
    {
        return passRecall;
    }
    if (hasDenyClass)
    {
        return denyRecall;
    }
    return 0.0;
}

std::string CsvEscape(const std::string& value)
{
    bool needsQuotes = value.find_first_of(",\"\n\r") != std::string::npos;
    if (!needsQuotes)
    {
        return value;
    }

    std::string escaped = "\"";
    for (char ch : value)
    {
        if (ch == '"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

template <typename Fn>
double MeasureMilliseconds(Fn&& fn)
{
    auto started = std::chrono::high_resolution_clock::now();
    fn();
    auto elapsed = std::chrono::high_resolution_clock::now() - started;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

std::string PathToString(const std::filesystem::path& path)
{
    return path.string();
}

std::string Trim(std::string value)
{
    auto isSpace = [](unsigned char ch) {
        return std::isspace(ch) != 0;
    };

    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) {
        return !isSpace(static_cast<unsigned char>(ch));
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) {
        return !isSpace(static_cast<unsigned char>(ch));
    }).base(), value.end());
    return value;
}

std::string NormalizeIdentity(std::string value)
{
    value = Trim(std::move(value));
    std::replace(value.begin(), value.end(), '_', ' ');
    while (value.find("  ") != std::string::npos)
    {
        value.replace(value.find("  "), 2, " ");
    }

    for (char& ch : value)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    return value;
}

std::string ExpectedNameFromFile(const std::filesystem::path& path)
{
    std::string stem = path.stem().string();
    while (!stem.empty() && std::isdigit(static_cast<unsigned char>(stem.back())))
    {
        stem.pop_back();
    }
    while (!stem.empty() && (stem.back() == '_' || stem.back() == '-' || stem.back() == ' '))
    {
        stem.pop_back();
    }

    return Trim(stem);
}

void AddDescriptor(
    std::map<AlgorithmId, cv::Mat>& sums,
    std::map<AlgorithmId, int>& counts,
    AlgorithmId algorithm,
    const cv::Mat& descriptor
)
{
    auto found = sums.find(algorithm);
    if (found == sums.end())
    {
        sums[algorithm] = descriptor.clone();
    }
    else
    {
        found->second += descriptor;
    }
    ++counts[algorithm];
}

std::vector<AlgorithmId> CpuAlgorithms()
{
    return {AlgorithmId::CustomCpu, AlgorithmId::OpenCvCpu};
}

std::vector<AlgorithmId> CudaAlgorithms()
{
    return {AlgorithmId::CustomCuda, AlgorithmId::OpenCvCuda};
}

bool ContainsAlgorithm(const std::vector<AlgorithmId>& algorithms, AlgorithmId id)
{
    return std::find(algorithms.begin(), algorithms.end(), id) != algorithms.end();
}

std::vector<AlgorithmId> RequestedAlgorithms(const std::vector<AlgorithmId>& selectedAlgorithms)
{
    if (selectedAlgorithms.empty())
    {
        std::vector<AlgorithmId> all;
        for (const AlgorithmInfo& algorithm : Algorithms())
        {
            all.push_back(algorithm.id);
        }
        return all;
    }

    std::vector<AlgorithmId> requested;
    for (const AlgorithmInfo& algorithm : Algorithms())
    {
        if (ContainsAlgorithm(selectedAlgorithms, algorithm.id))
        {
            requested.push_back(algorithm.id);
        }
    }
    return requested;
}

std::vector<AlgorithmId> FilterAlgorithmsByCuda(const std::vector<AlgorithmId>& algorithms, bool usesCuda)
{
    std::vector<AlgorithmId> filtered;
    for (AlgorithmId algorithm : algorithms)
    {
        if (GetAlgorithmInfo(algorithm).usesCuda == usesCuda)
        {
            filtered.push_back(algorithm);
        }
    }
    return filtered;
}

double ThresholdFor(AlgorithmId algorithm, const RecognitionThresholds& thresholds)
{
    switch (algorithm)
    {
    case AlgorithmId::CustomCpu:
    case AlgorithmId::CustomCuda:
        return thresholds.custom;
    case AlgorithmId::OpenCvCpu:
    case AlgorithmId::OpenCvCuda:
        return thresholds.openCv;
    }

    return 1.0;
}

bool DatabaseHasIdentity(const FaceDatabase& database, const std::string& expectedName)
{
    std::string expected = NormalizeIdentity(expectedName);
    return std::any_of(database.Records().begin(), database.Records().end(), [&](const FaceRecord& record) {
        return NormalizeIdentity(record.name) == expected;
    });
}

bool SameIdentity(const std::string& first, const std::string& second)
{
    return NormalizeIdentity(first) == NormalizeIdentity(second);
}
} // namespace

RecognitionEngine::RecognitionEngine(std::filesystem::path modelPath, std::filesystem::path databaseRoot)
    : _detector(modelPath.string()),
      _database(std::move(databaseRoot))
{
    _database.Load();
}

void RecognitionEngine::ReloadDatabase()
{
    _database.Load();
}

void RecognitionEngine::EnrollFace(
    const std::string& name,
    const std::vector<std::filesystem::path>& imagePaths,
    const ProgressCallback& progress
)
{
    if (name.empty())
    {
        throw std::invalid_argument("Введите имя лица перед добавлением");
    }
    if (imagePaths.empty())
    {
        throw std::invalid_argument("Выберите хотя бы одно изображение");
    }

    std::map<AlgorithmId, cv::Mat> sums;
    std::map<AlgorithmId, int> counts;
    int accepted = 0;
    int skipped = 0;

    for (const std::filesystem::path& path : imagePaths)
    {
        try
        {
            cv::Mat crop = LoadAndCropFace(path);
            cv::Mat cpuGray = PreprocessCpu(crop);
            std::map<AlgorithmId, cv::Mat> imageDescriptors;

            imageDescriptors[AlgorithmId::CustomCpu] = ComputeDescriptor(AlgorithmId::CustomCpu, cpuGray);
            imageDescriptors[AlgorithmId::OpenCvCpu] = ComputeDescriptor(AlgorithmId::OpenCvCpu, cpuGray);

            if (IsCudaAvailable())
            {
                try
                {
                    cv::Mat cudaGray = PreprocessCuda(crop);
                    imageDescriptors[AlgorithmId::CustomCuda] = ComputeDescriptor(AlgorithmId::CustomCuda, cudaGray);
                    imageDescriptors[AlgorithmId::OpenCvCuda] = ComputeDescriptor(AlgorithmId::OpenCvCuda, cudaGray);
                }
                catch (const std::exception& error)
                {
                    if (progress)
                    {
                        progress("CUDA-признаки пропущены для " + PathToString(path.filename()) + ": " + error.what());
                    }
                }
            }

            for (const auto& [algorithm, descriptor] : imageDescriptors)
            {
                AddDescriptor(sums, counts, algorithm, descriptor);
            }
            ++accepted;
            if (progress)
            {
                progress("Принято: " + PathToString(path.filename()));
            }
        }
        catch (const std::exception& error)
        {
            ++skipped;
            if (progress)
            {
                progress("Пропущено " + PathToString(path.filename()) + ": " + error.what());
            }
        }
    }

    if (accepted == 0)
    {
        throw std::runtime_error("Не найдено ни одного пригодного изображения лица");
    }

    FaceRecord record;
    record.name = name;
    record.sampleCount = accepted;
    for (auto& [algorithm, descriptor] : sums)
    {
        descriptor /= static_cast<float>(counts[algorithm]);
        record.descriptors[algorithm] = descriptor;
    }

    _database.Save(record);
    _database.Load();

    if (progress)
    {
        progress("Лицо '" + name + "' сохранено по " + std::to_string(accepted) +
                 " фото, пропущено " + std::to_string(skipped) + ".");
        progress("Лиц в базе: " + std::to_string(_database.Records().size()));
    }
}

SearchReport RecognitionEngine::SearchDirectory(
    const std::filesystem::path& directory,
    const ProgressCallback& progress,
    const RecognitionThresholds& thresholds,
    const std::vector<AlgorithmId>& selectedAlgorithms
)
{
    const std::vector<AlgorithmId> requestedAlgorithms = RequestedAlgorithms(selectedAlgorithms);
    if (requestedAlgorithms.empty())
    {
        throw std::invalid_argument("Выберите хотя бы один алгоритм для проверки.");
    }

    const std::vector<AlgorithmId> cpuAlgorithms = FilterAlgorithmsByCuda(requestedAlgorithms, false);
    const std::vector<AlgorithmId> cudaAlgorithms = FilterAlgorithmsByCuda(requestedAlgorithms, true);
    if (cpuAlgorithms.empty() && (!IsCudaAvailable() || cudaAlgorithms.empty()))
    {
        throw std::runtime_error("Нет доступных выбранных алгоритмов. CUDA-алгоритмы требуют доступную CUDA.");
    }

    _database.Load();
    if (_database.Records().empty())
    {
        throw std::runtime_error("База лиц пуста. Сначала добавьте хотя бы одно лицо.");
    }

    std::vector<CandidateFace> candidates = LoadCandidates(directory, progress);
    if (candidates.empty())
    {
        throw std::runtime_error("В выбранной папке не найдено изображений с распознанными лицами.");
    }

    if (progress)
    {
        progress("Запуск выбранных алгоритмов для " + std::to_string(candidates.size()) + " лиц.");
        progress("Пороги: моя LBPH=" + FormatDouble(thresholds.custom, 4) +
                 ", OpenCV LBPH=" + FormatDouble(thresholds.openCv, 4) + ".");
    }

    SearchReport report;
    std::mutex reportMutex;

    auto runGroup = [&](std::vector<AlgorithmId> algorithms) {
        for (AlgorithmId algorithm : algorithms)
        {
            std::vector<ImageAlgorithmResult> localResults;
            AlgorithmRunSummary summary = RunAlgorithm(algorithm, candidates, localResults, progress, thresholds);

            std::lock_guard lock(reportMutex);
            report.summaries.push_back(summary);
            report.results.insert(report.results.end(), localResults.begin(), localResults.end());
        }
    };

    std::future<void> cpuBranch;
    if (!cpuAlgorithms.empty())
    {
        cpuBranch = std::async(std::launch::async, runGroup, cpuAlgorithms);
    }

    std::future<void> cudaBranch;
    if (!cudaAlgorithms.empty() && IsCudaAvailable())
    {
        cudaBranch = std::async(std::launch::async, runGroup, cudaAlgorithms);
    }

    if (cpuBranch.valid())
    {
        cpuBranch.get();
    }

    if (cudaBranch.valid())
    {
        cudaBranch.get();
    }
    else if (!cudaAlgorithms.empty() && progress)
    {
        progress("CUDA недоступна: GPU-алгоритмы пропущены.");
    }

    if (progress)
    {
        for (const AlgorithmRunSummary& summary : report.summaries)
        {
            progress(AlgorithmTitle(summary.algorithm) + ": " +
                     FormatDouble(summary.milliseconds, 2) + " мс, обработано " +
                     std::to_string(summary.processed) + ", сбоев " + std::to_string(summary.failed) +
                     ", пропуск=" + std::to_string(summary.passed) +
                     ", отказ=" + std::to_string(summary.denied) +
                     ", успех=" + std::to_string(summary.correct) +
                     ", ошибок=" + std::to_string(summary.errors) +
                     ", сбалансированная точность=" + FormatDouble(BalancedAccuracyPercent(summary), 2) + "%" +
                     ", среднее=" + FormatDouble(summary.averageMilliseconds, 2) + " мс/лицо" +
                     ", лучший score=" + FormatDouble(summary.bestSimilarity));
        }

        for (const ImageAlgorithmResult& result : report.results)
        {
            if (!result.error.empty())
            {
                progress(PathToString(result.imagePath.filename()) + " | " +
                         AlgorithmTitle(result.algorithm) + " | сбой: " + result.error);
                continue;
            }

            progress(PathToString(result.imagePath.filename()) + " | " +
                     AlgorithmTitle(result.algorithm) + " -> " +
                     (result.recognized ? "ПРОПУСК " : "ОТКАЗ, лучший кандидат ") +
                     (result.match.found ? result.match.name : "none") +
                     " | ожидалось=" + result.expectedName +
                     " | итог=" + result.outcome +
                     (result.errorType.empty() ? "" : " | ошибка=" + result.errorType) +
                     " | score=" + FormatDouble(result.match.similarity) +
                     " | порог=" + FormatDouble(result.threshold) +
                     " | время=" + FormatDouble(result.totalMilliseconds, 2) + " мс");
        }
    }

    return report;
}

void RecognitionEngine::SaveReportCsv(const SearchReport& report, const std::filesystem::path& path) const
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("Не удалось записать CSV-отчет");
    }

    output << "section,algorithm,image,expected,expected_in_db,verdict,best_match,similarity,threshold,outcome,error_type,total_ms,processed,failed,pass,deny,correct,errors,expected_pass,expected_deny,correct_pass,balanced_accuracy,pass_recall,deny_recall,false_accept,false_reject,wrong_match,correct_deny,avg_ms,best_similarity,error\n";

    for (const AlgorithmRunSummary& summary : report.summaries)
    {
        output
            << "summary,"
            << CsvEscape(AlgorithmTitle(summary.algorithm)) << ",,,,,,,,,,"
            << FormatDouble(summary.milliseconds, 6) << ","
            << summary.processed << ","
            << summary.failed << ","
            << summary.passed << ","
            << summary.denied << ","
            << summary.correct << ","
            << summary.errors << ","
            << summary.expectedPass << ","
            << summary.expectedDeny << ","
            << summary.correctPasses << ","
            << FormatDouble(BalancedAccuracyPercent(summary), 6) << ","
            << FormatDouble(ClassRecallPercent(summary.correctPasses, summary.expectedPass), 6) << ","
            << FormatDouble(ClassRecallPercent(summary.correctDenies, summary.expectedDeny), 6) << ","
            << summary.falseAccepts << ","
            << summary.falseRejects << ","
            << summary.wrongMatches << ","
            << summary.correctDenies << ","
            << FormatDouble(summary.averageMilliseconds, 6) << ","
            << FormatDouble(summary.bestSimilarity, 6) << ",\n";
    }

    for (const ImageAlgorithmResult& result : report.results)
    {
        output
            << "image,"
            << CsvEscape(AlgorithmTitle(result.algorithm)) << ","
            << CsvEscape(result.imagePath.string()) << ","
            << CsvEscape(result.expectedName) << ","
            << (result.expectedInDatabase ? "yes" : "no") << ","
            << (result.recognized ? "PASS" : "DENY") << ","
            << CsvEscape(result.match.found ? result.match.name : "") << ","
            << FormatDouble(result.match.similarity, 6) << ","
            << FormatDouble(result.threshold, 6) << ","
            << CsvEscape(result.outcome) << ","
            << CsvEscape(result.errorType) << ","
            << FormatDouble(result.totalMilliseconds, 6) << ","
            << std::string(17, ',')
            << CsvEscape(result.error) << "\n";
    }
}

int RecognitionEngine::KnownFacesCount() const
{
    return static_cast<int>(_database.Records().size());
}

const FaceDatabase& RecognitionEngine::Database() const noexcept
{
    return _database;
}

cv::Mat RecognitionEngine::LoadAndCropFace(const std::filesystem::path& path)
{
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty())
    {
        throw std::runtime_error("Не удалось прочитать изображение");
    }

    cv::Mat crop = _detector.Detect(image);
    if (crop.empty())
    {
        throw std::runtime_error("Лицо не обнаружено");
    }

    return crop;
}

std::vector<RecognitionEngine::CandidateFace> RecognitionEngine::LoadCandidates(
    const std::filesystem::path& directory,
    const ProgressCallback& progress
)
{
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory))
    {
        throw std::invalid_argument("Выбранный путь не является папкой");
    }

    std::vector<CandidateFace> candidates;
    int skipped = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file() || !IsImageFile(entry.path()))
        {
            continue;
        }

        try
        {
            std::string expectedName = ExpectedNameFromFile(entry.path());
            candidates.push_back({
                entry.path(),
                expectedName,
                DatabaseHasIdentity(_database, expectedName),
                LoadAndCropFace(entry.path())
            });
        }
        catch (const std::exception& error)
        {
            ++skipped;
            if (progress)
            {
                progress("Пропущено " + PathToString(entry.path().filename()) + ": " + error.what());
            }
        }
    }

    if (progress)
    {
        progress("Подготовлено лиц: " + std::to_string(candidates.size()) +
                 ", пропущено файлов: " + std::to_string(skipped) + ".");
    }

    return candidates;
}

AlgorithmRunSummary RecognitionEngine::RunAlgorithm(
    AlgorithmId algorithm,
    const std::vector<CandidateFace>& candidates,
    std::vector<ImageAlgorithmResult>& results,
    const ProgressCallback& progress,
    const RecognitionThresholds& thresholds
) const
{
    AlgorithmRunSummary summary;
    summary.algorithm = algorithm;

    auto started = std::chrono::high_resolution_clock::now();
    bool useCudaPreprocess = GetAlgorithmInfo(algorithm).usesCuda;

    int index = 0;
    for (const CandidateFace& candidate : candidates)
    {
        ++index;
        ImageAlgorithmResult result;
        result.imagePath = candidate.path;
        result.expectedName = candidate.expectedName;
        result.expectedInDatabase = candidate.expectedInDatabase;
        result.algorithm = algorithm;
        result.threshold = ThresholdFor(algorithm, thresholds);
        if (candidate.expectedInDatabase)
        {
            ++summary.expectedPass;
        }
        else
        {
            ++summary.expectedDeny;
        }

        try
        {
            result.totalMilliseconds = MeasureMilliseconds([&] {
                cv::Mat prepared;
                prepared = useCudaPreprocess ? PreprocessCuda(candidate.crop) : PreprocessCpu(candidate.crop);

                cv::Mat descriptor;
                descriptor = ComputeDescriptor(algorithm, prepared);

                result.match = _database.FindBest(algorithm, descriptor);
            });
            result.recognized = result.match.found && result.match.similarity >= result.threshold;
            summary.bestSimilarity = std::max(summary.bestSimilarity, result.match.similarity);

            if (result.recognized)
            {
                ++summary.passed;
                if (candidate.expectedInDatabase && SameIdentity(candidate.expectedName, result.match.name))
                {
                    result.correct = true;
                    result.outcome = "УСПЕХ";
                    result.errorType = "";
                    ++summary.correct;
                    ++summary.correctPasses;
                }
                else if (candidate.expectedInDatabase)
                {
                    result.outcome = "ОШИБКА";
                    result.errorType = "НЕ_ТО_ЛИЦО";
                    ++summary.errors;
                    ++summary.wrongMatches;
                }
                else
                {
                    result.outcome = "ОШИБКА";
                    result.errorType = "ЛОЖНЫЙ_ПРОПУСК";
                    ++summary.errors;
                    ++summary.falseAccepts;
                }
            }
            else
            {
                ++summary.denied;
                if (candidate.expectedInDatabase)
                {
                    result.outcome = "ОШИБКА";
                    result.errorType = "ЛОЖНЫЙ_ОТКАЗ";
                    ++summary.errors;
                    ++summary.falseRejects;
                }
                else
                {
                    result.correct = true;
                    result.outcome = "УСПЕХ";
                    result.errorType = "";
                    ++summary.correct;
                    ++summary.correctDenies;
                }
            }
            ++summary.processed;

            if (progress)
            {
                progress("[" + AlgorithmTitle(algorithm) + "] " +
                         std::to_string(index) + "/" + std::to_string(candidates.size()) +
                         " " + PathToString(candidate.path.filename()) +
                         " ожидалось=" + candidate.expectedName +
                         " решение=" + std::string(result.recognized ? "ПРОПУСК" : "ОТКАЗ") +
                         " найдено=" + (result.match.found ? result.match.name : "none") +
                         " итог=" + result.outcome +
                         (result.errorType.empty() ? "" : " ошибка=" + result.errorType) +
                         " score=" + FormatDouble(result.match.similarity) +
                         " порог=" + FormatDouble(result.threshold) +
                         " время=" + FormatDouble(result.totalMilliseconds, 2) + " мс");
            }
        }
        catch (const std::exception& error)
        {
            result.error = error.what();
            ++summary.failed;
            if (progress)
            {
                progress("[" + AlgorithmTitle(algorithm) + "] " +
                         std::to_string(index) + "/" + std::to_string(candidates.size()) +
                         " сбой: " + error.what());
            }
        }

        results.push_back(std::move(result));
    }

    auto elapsed = std::chrono::high_resolution_clock::now() - started;
    summary.milliseconds = std::chrono::duration<double, std::milli>(elapsed).count();
    if (summary.processed > 0)
    {
        summary.averageMilliseconds = summary.milliseconds / static_cast<double>(summary.processed);
    }
    return summary;
}
} // namespace biometrics
