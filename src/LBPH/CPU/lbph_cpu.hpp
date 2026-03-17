#ifndef LBPH_CPU_HPP
#define LBPH_CPU_HPP

#include "opencv2/opencv.hpp"

namespace LBPH::cpu
{
class LBPH
{
public:

    LBPH(std::unique_ptr<cv::Mat> img_);
    // ~LBPH();

    /**
     * @brief Сравнение LBPH
     */
    static double Distance(const LBPH& standart, const LBPH& tested);
private:
    /**
     * @brief Вычисление LBP кодов радиусов 1, 2 и 3
     */
    void LBPCodes() noexcept;

    /**
     * @brief Нормализация гистограмм
     */
    void NormalizeHistogram();

    /**
     * @brief Вычисление LBP кодов изображения с радиусом
     */
    cv::Mat LBPCode(int radius_) noexcept;

    /**
     * @brief Построение гистограммы
     */
    cv::Mat Histogram();

    /**
     * @brief Предобработка изображения
     */
    void PreprocessImage(cv::Mat& img_) noexcept;

    std::unique_ptr<cv::Mat> _img = nullptr;
    cv::Mat _normalizeHistogram = cv::Mat::zeros(256, 3, CV_64FC1);
    cv::Mat _LBPCode_r1 = cv::Mat::zeros(cv::Size(_img.get()->size()), CV_8UC1);
    cv::Mat _LBPCode_r2 = cv::Mat::zeros(cv::Size(_img.get()->size()), CV_8UC1);
    cv::Mat _LBPCode_r3 = cv::Mat::zeros(cv::Size(_img.get()->size()), CV_8UC1);
};
} // namespace LBPH::cpu

#endif