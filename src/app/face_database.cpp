#include "app/face_database.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "opencv2/core.hpp"

namespace biometrics
{
namespace
{
std::string SafeFileName(std::string value)
{
    for (char& ch : value)
    {
        unsigned char c = static_cast<unsigned char>(ch);
        if (!std::isalnum(c) && ch != '-' && ch != '_')
        {
            ch = '_';
        }
    }

    value.erase(
        std::remove_if(value.begin(), value.end(), [](char ch) {
            return ch == '\0';
        }),
        value.end()
    );

    if (value.empty())
    {
        value = "face";
    }

    return value;
}
} // namespace

FaceDatabase::FaceDatabase(std::filesystem::path root)
    : _root(std::move(root))
{
}

void FaceDatabase::Load()
{
    _records.clear();
    std::filesystem::create_directories(_root);

    for (const auto& entry : std::filesystem::directory_iterator(_root))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".yml")
        {
            continue;
        }

        cv::FileStorage storage(entry.path().string(), cv::FileStorage::READ);
        if (!storage.isOpened())
        {
            continue;
        }

        FaceRecord record;
        storage["name"] >> record.name;
        storage["sample_count"] >> record.sampleCount;
        cv::FileNode aggregationNode = storage["aggregation"];
        if (!aggregationNode.empty())
        {
            aggregationNode >> record.aggregation;
        }
        if (record.aggregation.empty())
        {
            record.aggregation = "mean_descriptor";
        }

        for (const AlgorithmInfo& algorithm : Algorithms())
        {
            cv::Mat descriptor;
            storage[algorithm.key] >> descriptor;
            if (!descriptor.empty())
            {
                if (descriptor.type() != CV_32FC1)
                {
                    descriptor.convertTo(descriptor, CV_32FC1);
                }
                record.descriptors[algorithm.id] = descriptor.reshape(1, 1).clone();

                int descriptorCount = 0;
                cv::FileNode countNode = storage[std::string(algorithm.key) + "_count"];
                if (!countNode.empty())
                {
                    countNode >> descriptorCount;
                }
                if (descriptorCount <= 0)
                {
                    descriptorCount = std::max(record.sampleCount, 1);
                }
                record.descriptorCounts[algorithm.id] = descriptorCount;
            }
        }

        if (!record.name.empty() && !record.descriptors.empty())
        {
            _records.push_back(std::move(record));
        }
    }
}

void FaceDatabase::Save(const FaceRecord& record) const
{
    if (record.name.empty())
    {
        throw std::invalid_argument("Имя лица не должно быть пустым");
    }

    std::filesystem::create_directories(_root);
    cv::FileStorage storage(RecordPath(record.name).string(), cv::FileStorage::WRITE);
    if (!storage.isOpened())
    {
        throw std::runtime_error("Не удалось открыть файл базы для записи");
    }

    storage << "name" << record.name;
    storage << "sample_count" << record.sampleCount;
    storage << "aggregation" << record.aggregation;
    for (const AlgorithmInfo& algorithm : Algorithms())
    {
        auto found = record.descriptors.find(algorithm.id);
        if (found != record.descriptors.end() && !found->second.empty())
        {
            storage << algorithm.key << found->second;
            auto count = record.descriptorCounts.find(algorithm.id);
            int descriptorCount = count != record.descriptorCounts.end()
                ? std::max(count->second, 1)
                : std::max(record.sampleCount, 1);
            storage << (std::string(algorithm.key) + "_count") << descriptorCount;
        }
    }
}

MatchResult FaceDatabase::FindBest(AlgorithmId algorithm, const cv::Mat& descriptor) const
{
    MatchResult best;
    for (const FaceRecord& record : _records)
    {
        auto found = record.descriptors.find(algorithm);
        if (found == record.descriptors.end() || found->second.empty())
        {
            continue;
        }

        double similarity = CompareDescriptors(algorithm, found->second, descriptor);
        if (!best.found || similarity > best.similarity)
        {
            best.found = true;
            best.name = record.name;
            best.similarity = similarity;
        }
    }

    return best;
}

const std::vector<FaceRecord>& FaceDatabase::Records() const noexcept
{
    return _records;
}

const std::filesystem::path& FaceDatabase::Root() const noexcept
{
    return _root;
}

std::filesystem::path FaceDatabase::RecordPath(const std::string& name) const
{
    return _root / (SafeFileName(name) + ".yml");
}
} // namespace biometrics
