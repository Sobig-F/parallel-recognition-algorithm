#include <math.h>
#include <numbers>

#include "lbph_cpu.hpp"

namespace LBPH::cpu
{
int LBPH::_i = 0;
LBPH::LBPH(const cv::Mat img_)
: _img(img_)
{
    LBPCodes();
    NormalizeHistogram();
}

void LBPH::LBPCodes() noexcept
{
    ++_i;
    _LBPCode_r1 = LBPH::LBPCode(1);
    _LBPCode_r2 = LBPH::LBPCode(2);
    _LBPCode_r3 = LBPH::LBPCode(3);
    cv::Mat lbp_vis;
    _LBPCode_r1.convertTo(lbp_vis, CV_8U);  // 0-255 → 0-255 (уже в нужном диапазоне)
    cv::imwrite(SRC_DIR"/../" + std::to_string(_i) + ".png", lbp_vis);

    // std::cout << "LBP Code stats:" << std::endl;

    // auto printStats = [](const cv::Mat& lbp, const std::string& label) {
    //     cv::Mat hist(256, 1, CV_32S, cv::Scalar(0));
    //     for (int r = 0; r < lbp.rows; ++r) {
    //         for (int c = 0; c < lbp.cols; ++c) {
    //             hist.at<int>(lbp.at<uchar>(r, c), 0)++;
    //         }
    //     }
        
    //     int maxBin = 0, maxVal = 0;
    //     for (int i = 0; i < 256; ++i) {
    //         if (hist.at<int>(i, 0) > maxVal) {
    //             maxVal = hist.at<int>(i, 0);
    //             maxBin = i;
    //         }
    //     }
        
    //     std::cout << label << ": max bin=" << maxBin 
    //             << " (count=" << maxVal << ")" << std::endl;
    // };

    // printStats(_LBPCode_r1, "R=1");
    // printStats(_LBPCode_r2, "R=2");
    // printStats(_LBPCode_r3, "R=3");
}

cv::Mat LBPH::LBPCode(int radius) noexcept {
    cv::Mat result = cv::Mat::zeros(_img.size(), CV_8UC1);
    
    // 8 углов для 8 соседей (в радианах)
    const float angles[8] = {
        0,
        -std::numbers::pi_v<float> / 4,
        -std::numbers::pi_v<float> / 2,
        -3 * std::numbers::pi_v<float> / 4,
        std::numbers::pi_v<float>,
        3 * std::numbers::pi_v<float> / 4,
        std::numbers::pi_v<float> / 2,
        std::numbers::pi_v<float> / 4
    };
    
    // Проходим только по пикселям, где все соседи помещаются в изображение
    for (int row = radius; row < _img.rows - radius; ++row) {
        for (int col = radius; col < _img.cols - radius; ++col) {
            
            uchar center = _img.at<uchar>(row, col);
            uchar lbp_code = 0;
            
            // Проверяем 8 соседей на окружности
            for (int p = 0; p < 8; ++p) {
                // Координаты соседа (плавающие)
                float x = col + radius * std::cos(angles[p]);
                float y = row + radius * std::sin(angles[p]);
                
                // === БИЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ ===
                int x0 = static_cast<int>(std::floor(x));  // col
                int y0 = static_cast<int>(std::floor(y));  // row
                int x1 = std::min(x0 + 1, _img.cols - 1);
                int y1 = std::min(y0 + 1, _img.rows - 1);
                x0 = std::max(x0, 0);
                y0 = std::max(y0, 0);

                float dx = x - x0;  // дробная часть по X
                float dy = y - y0;  // дробная часть по Y

                // Доступ к пикселям: at<uchar>(ROW, COL) = at<uchar>(y, x)
                uchar p00 = _img.at<uchar>(y0, x0);  // верх-лево
                uchar p01 = _img.at<uchar>(y0, x1);  // верх-право
                uchar p10 = _img.at<uchar>(y1, x0);  // низ-лево
                uchar p11 = _img.at<uchar>(y1, x1);  // низ-право

                float interpolated = 
                    p00 * (1-dx) * (1-dy) +  // верх-лево
                    p01 * dx * (1-dy) +      // верх-право
                    p10 * (1-dx) * dy +      // низ-лево
                    p11 * dx * dy;           // низ-право

                // Сравниваем с центром
                if (interpolated >= center) {
                    lbp_code |= (1 << p);
                }
            }
            
            result.at<uchar>(row, col) = lbp_code;
        }
    }
    
    return result;
}

cv::Mat LBPH::Histogram()
{
    cv::Mat result(256, 3, CV_64FC1, cv::Scalar(0.0));

    for (int row = 0; row < _LBPCode_r1.rows; ++row)
    {
        for (int col = 0; col < _LBPCode_r1.cols; ++col)
        {
            result.ptr<double>(_LBPCode_r1.ptr<uchar>(row)[col])[0] += 1.0;
        }
    }
    for (int row = 0; row < _LBPCode_r2.rows; ++row)
    {
        for (int col = 0; col < _LBPCode_r2.cols; ++col)
        {
            result.ptr<double>(_LBPCode_r2.ptr<uchar>(row)[col])[1] += 1.0;
        }
    }
    for (int row = 0; row < _LBPCode_r3.rows; ++row)
    {
        for (int col = 0; col < _LBPCode_r3.cols; ++col)
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

double LBPH::Similarity(const LBPH& standart, const LBPH& tested)
{
    std::vector<double> w = {0.4, 0.3, 0.3};
    double alpha = 0.6, beta = 0.4;

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