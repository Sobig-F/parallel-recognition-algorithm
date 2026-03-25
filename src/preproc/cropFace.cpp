#include "cropFace.hpp"

namespace preprocessing
{
FaceDetector::FaceDetector(const std::string& modelPath_)
:   _confidenceThreshold{0.8f},
    _init{false}
{
    if (!cv::utils::fs::exists(modelPath_))
    {
        return;
    }

    try
    {
        _net = cv::FaceDetectorYN::create(
            modelPath_,
            "",
            cv::Size(BLOBX, BLOBY),
            _confidenceThreshold,
            0.3,
            5000
        );

        if (_net.empty())
        {
            std::cerr << "Failed to load model" << std::endl;
            return;
        }

        _init = true;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}

cv::Mat FaceDetector::Detect(const cv::Mat& input_)
{
    cv::Size originalSize = input_.size();

    _net->setInputSize(originalSize);
    cv::Mat net_outputs;
    _net->detect(input_, net_outputs);

    std::vector<cv::Rect> faces;

    for (int i = 0; i < net_outputs.rows; ++i)
    {
        cv::Rect rect = {
            static_cast<int>(std::max(0, static_cast<int>(net_outputs.at<float>(i, 0)))),
            static_cast<int>(std::max(0, static_cast<int>(net_outputs.at<float>(i, 1)))),
            static_cast<int>(net_outputs.at<float>(i, 2)),
            static_cast<int>(net_outputs.at<float>(i, 3)),
        };

        rect.width = std::min(originalSize.width - rect.x, rect.width);
        rect.height = std::min(originalSize.height - rect.y, rect.height);
        
        faces.push_back(rect);
    }

    if (faces.empty())
    {
        return cv::Mat();
    }

    int bigges_area_index = 0;
    int faces_count = faces.size();
    for (int i = 1; i < faces_count; ++i)
    {
        if (faces[i].area() > faces[bigges_area_index].area())
        {
            bigges_area_index = i;
        }
    }
    
    return input_(faces[bigges_area_index]).clone();
}

void GrayScale(cv::Mat& input_)
{
    cv::Mat result;
    if (input_.channels() == 3) {
        cv::cvtColor(input_, result, cv::COLOR_BGR2GRAY);
    } else if (input_.channels() == 4) {
        cv::cvtColor(input_, result, cv::COLOR_BGRA2GRAY);
    } else if (input_.channels() == 1) {
        result = input_;
    }

    input_ = std::move(result);
}

void GaussainBlur(cv::Mat& input_)
{
    cv::Mat result;

    cv::GaussianBlur(
        input_,
        result,
        cv::Size(5, 5),
        1.0,
        1.0,
        cv::BORDER_DEFAULT
    );

    input_ = std::move(result);
}

void CLAHE(cv::Mat& input_)
{
    cv::Mat result;

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        2.0,
        cv::Size(8, 8)
    );

    clahe->apply(input_, result);

    input_ = std::move(result);
}

void Resize(cv::Mat& input_)
{
    cv::Mat result;
    cv::resize(input_, result, cv::Size(TARGET_X, TARGET_Y), 0, 0, cv::INTER_LINEAR);
    input_ = std::move(result);
}

} // namespace preprocessing