#include "preproc.hpp"

namespace preprocessing
{
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
        cv::Size(3, 3),
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
        3.0,
        cv::Size(9, 9)
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
}