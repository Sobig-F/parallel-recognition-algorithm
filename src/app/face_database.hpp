#ifndef FACE_DATABASE_HPP
#define FACE_DATABASE_HPP

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "LBPH/lbph_descriptors.hpp"

namespace biometrics
{
struct FaceRecord
{
    std::string name;
    int sampleCount = 0;
    std::map<AlgorithmId, cv::Mat> descriptors;
};

struct MatchResult
{
    std::string name;
    double similarity = 0.0;
    bool found = false;
};

class FaceDatabase
{
public:
    explicit FaceDatabase(std::filesystem::path root);

    void Load();
    void Save(const FaceRecord& record) const;
    MatchResult FindBest(AlgorithmId algorithm, const cv::Mat& descriptor) const;

    const std::vector<FaceRecord>& Records() const noexcept;
    const std::filesystem::path& Root() const noexcept;

private:
    std::filesystem::path RecordPath(const std::string& name) const;

    std::filesystem::path _root;
    std::vector<FaceRecord> _records;
};
} // namespace biometrics

#endif
