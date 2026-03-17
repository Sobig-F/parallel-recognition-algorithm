// #include "lbph_cpu.hpp"
// // #include "opencv2/opencv.hpp"

// cv::Mat LBPH::cpu::PreprocessImage(const cv::Mat& img_) noexcept
// {
//     cv::Mat img_gray, img_equalized, result;
//     if (img_.channels() == 3) {
//         cv::cvtColor(img_, img_gray, cv::COLOR_BGR2GRAY);
//     } else if (img_.channels() == 4) {
//         cv::cvtColor(img_, img_gray, cv::COLOR_BGRA2GRAY);
//     } else if (img_.channels() == 1) {
//         img_gray = img_;
//     }

//     cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
//     clahe->apply(img_gray, img_equalized);

//     cv::resize(img_equalized, result, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);

//     return result;
// }

// cv::Mat LBPH::cpu::LBPCode(const cv::Mat& img_, LBP_Radius radius) noexcept
// {
//     cv::Mat result = cv::Mat::zeros(cv::Size(img_.size()), CV_8UC1);
//     int local_lbp = 0;

//     if (radius == 1)
//     {
//         for (int row = 1; row + 1 < img_.rows; ++row)
//         {
//             for (int col = 1; col + 1 < img_.cols; ++col)
//             {
//                 uchar center = img_.at<uchar>(row, col);
//                 if (img_.ptr<uchar>(row - 1)[col - 1] > center) {
//                     local_lbp |= (1 << 0);
//                 }
//                 if (img_.ptr<uchar>(row - 1)[col] > center) {
//                     local_lbp |= (1 << 1);
//                 }
//                 if (img_.ptr<uchar>(row - 1)[col + 1] > center) {
//                     local_lbp |= (1 << 2);
//                 }
//                 if (img_.ptr<uchar>(row)[col + 1] > center) {
//                     local_lbp |= (1 << 3);
//                 }
//                 if (img_.ptr<uchar>(row + 1)[col + 1] > center) {
//                     local_lbp |= (1 << 4);
//                 }
//                 if (img_.ptr<uchar>(row + 1)[col] > center) {
//                     local_lbp |= (1 << 5);
//                 }
//                 if (img_.ptr<uchar>(row + 1)[col - 1] > center) {
//                     local_lbp |= (1 << 6);
//                 }
//                 if (img_.ptr<uchar>(row)[col - 1] > center) {
//                     local_lbp |= (1 << 7);
//                 }
                
//                 result.at<uchar>(row, col) = local_lbp;
//                 local_lbp = 0;
//             }
//         }
//     }

//     return result;
// }

// cv::Mat LBPH::cpu::Histogram(const cv::Mat& lbpcodes_)
// {
//     cv::Mat result(256, 1, CV_64FC1, cv::Scalar(0.0));

//     for (int row = 1; row + 1 < lbpcodes_.rows; ++row)
//     {
//         for (int col = 1; col + 1 < lbpcodes_.cols; ++col)
//         {
//             result.ptr<double>(lbpcodes_.ptr<uchar>(row)[col])[0] += 1.0;
//         }
//     }

//     return result;
// }

// void LBPH::cpu::NormalizeHistoram(cv::Mat& histogram_)
// {
//     double sum = cv::sum(histogram_)[0];
//     if (sum > 0)
//     {
//         histogram_ /= sum;
//     }
// }

// double LBPH::cpu::Distance(const cv::Mat& target, const cv::Mat& variant)
// {
//     double result = 0;
//     double a, b, diff;

//     for (int i = 0; i < 256; ++i)
//     {
//         a = target.ptr<double>(i)[0];
//         b = variant.ptr<double>(i)[0];

//         if (a + b > 0)
//         {
//             diff = a - b;
//             result += (diff * diff) / (a + b);
//         }
//     }

//     return result;
// }

#include <math.h>

#include "lbph_cpu.hpp"

namespace LBPH::cpu
{
LBPH::LBPH(std::unique_ptr<cv::Mat> img_)
: _img{move(img_)}
{
    PreprocessImage(*_img.get());
    LBPCodes();
    NormalizeHistogram();
}

void LBPH::PreprocessImage(cv::Mat& img_) noexcept
{
    cv::Mat img_gray, img_equalized, result;
    if (img_.channels() == 3) {
        cv::cvtColor(img_, img_gray, cv::COLOR_BGR2GRAY);
    } else if (img_.channels() == 4) {
        cv::cvtColor(img_, img_gray, cv::COLOR_BGRA2GRAY);
    } else if (img_.channels() == 1) {
        img_gray = img_;
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(img_gray, img_equalized);

    cv::resize(img_equalized, img_, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
}

void LBPH::LBPCodes() noexcept
{
    _LBPCode_r1 = LBPH::LBPCode(1);
    _LBPCode_r2 = LBPH::LBPCode(2);
    _LBPCode_r3 = LBPH::LBPCode(3);
}

cv::Mat LBPH::LBPCode(int radius_) noexcept
{
    cv::Mat result = cv::Mat::zeros(cv::Size(_img->size()), CV_8UC1);
    int local_lbp = 0;
    for (int row = radius_; row + radius_ <= _img->rows - radius_; ++row)
    {
        for (int col = radius_; col + radius_ <= _img->cols - radius_; ++col)
        {
            uchar center = _img->at<uchar>(row, col);
            if (_img->ptr<uchar>(row - radius_)[col - radius_] > center) {
                local_lbp |= (1 << 0);
            }
            if (_img->ptr<uchar>(row - radius_)[col] > center) {
                local_lbp |= (1 << 1);
            }
            if (_img->ptr<uchar>(row - radius_)[col + radius_] > center) {
                local_lbp |= (1 << 2);
            }
            if (_img->ptr<uchar>(row)[col + radius_] > center) {
                local_lbp |= (1 << 3);
            }
            if (_img->ptr<uchar>(row + radius_)[col + radius_] > center) {
                local_lbp |= (1 << 4);
            }
            if (_img->ptr<uchar>(row + radius_)[col] > center) {
                local_lbp |= (1 << 5);
            }
            if (_img->ptr<uchar>(row + radius_)[col - radius_] > center) {
                local_lbp |= (1 << 6);
            }
            if (_img->ptr<uchar>(row)[col - radius_] > center) {
                local_lbp |= (1 << 7);
            }
            
            result.at<uchar>(row, col) = local_lbp;
            local_lbp = 0;
        }
    }
    
    return result;
}

cv::Mat LBPH::Histogram()
{
    cv::Mat result(256, 3, CV_64FC1, cv::Scalar(0.0));

    for (int row = 1; row + 1 < _LBPCode_r1.rows; ++row)
    {
        for (int col = 1; col + 1 < _LBPCode_r1.cols; ++col)
        {
            result.ptr<double>(_LBPCode_r1.ptr<uchar>(row)[col])[0] += 1.0;
        }
    }
    for (int row = 1; row + 1 < _LBPCode_r2.rows; ++row)
    {
        for (int col = 1; col + 1 < _LBPCode_r2.cols; ++col)
        {
            result.ptr<double>(_LBPCode_r2.ptr<uchar>(row)[col])[1] += 1.0;
        }
    }
    for (int row = 1; row + 1 < _LBPCode_r3.rows; ++row)
    {
        for (int col = 1; col + 1 < _LBPCode_r3.cols; ++col)
        {
            result.ptr<double>(_LBPCode_r3.ptr<uchar>(row)[col])[2] += 1.0;
        }
    }

    return result;
}

void LBPH::NormalizeHistogram()
{
    _normalizeHistogram = Histogram();
    double sum = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        sum = cv::sum(_normalizeHistogram.col(i))[0];
        if (sum > 0.0)
        {
            _normalizeHistogram.col(i) /= sum;
        }
    }
}

double LBPH::Distance(const LBPH& standart, const LBPH& tested)
{
    std::vector<double> w = {0.2, 0.4, 0.4};
    double alpha = 0.8, beta = 0.2;

    double a, b, diff, dot_product, norm_A, norm_B, denominator, chi2_temp;
    double chi2 = 0.0, cosine = 0.0;

    for (int col = 0; col < 3; ++col)
    {
        chi2_temp = 0.0;
        dot_product = 0.0;
        norm_A = 0.0;
        norm_B = 0.0;

        for (int i = 0; i < 256; ++i)
        {
            a = standart._normalizeHistogram.ptr<double>(i)[col];
            b = tested._normalizeHistogram.ptr<double>(i)[col];

            dot_product += a * b;
            norm_A += a * a;
            norm_B += b * b;

            if (a + b > 0)
            {
                diff = a - b;
                chi2_temp += (diff * diff) / (a + b);
            }
        }

        norm_A = sqrt(norm_A);
        norm_B = sqrt(norm_B);
        denominator = norm_A * norm_B;
        if (denominator > 1e-10) {
            cosine += w[col] * (dot_product / denominator);
        }
        chi2 += w[col] * (1 / (1 + chi2_temp));
    }

    return alpha * chi2 + beta * cosine;
}

} // namespace LBPH::cpu