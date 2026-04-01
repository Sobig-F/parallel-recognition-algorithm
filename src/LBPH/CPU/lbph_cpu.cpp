#include <math.h>
#include <numbers>

#include "lbph_cpu.hpp"

static int CountTransitions(uchar code)
{
    int transitions = 0;
    for (int p = 0; p < 8; ++p)
    {
        int bit_curr = (code >> p) & 1;
        int bit_next = (code >> ((p + 1) % 8)) & 1;
        if (bit_curr != bit_next)
            ++transitions;
    }
    return transitions;
}

static uchar ToUniformCode(uchar code)
{
    if (CountTransitions(code) <= 2)
    {
        // Номер бина = число единичных битов (0..8), но бины 0..57 для uniform
        // Используем просто popcount как индекс (0..8) — 9 uniform бинов
        // Стандартная таблица: 59 бинов (0 единиц, 1 единица x8 позиций, ..., 8 единиц)
        // Для простоты возвращаем popcount (0-8)
        return static_cast<uchar>(std::popcount(code));
    }
    return 9; // non-uniform бин
}

namespace LBPH::cpu
{
int LBPH::_i = 0;

LBPH::LBPH(const cv::Mat img_)
: _img(img_)
{
    ++_i;
    LBPCodes();
    NormalizeHistogram();
}

void LBPH::LBPCodes() noexcept
{
    _LBPCode_r1 = LBPCode(1);
    _LBPCode_r2 = LBPCode(2);
    _LBPCode_r3 = LBPCode(3);
    // cv::Mat lbp_vis;
    // _LBPCode_r1.convertTo(lbp_vis, CV_8U);  // 0-255 → 0-255 (уже в нужном диапазоне)
    // cv::imwrite(SRC_DIR"/../" + std::to_string(_i) + "_R1.png", lbp_vis);
    // _LBPCode_r2.convertTo(lbp_vis, CV_8U);  // 0-255 → 0-255 (уже в нужном диапазоне)
    // cv::imwrite(SRC_DIR"/../" + std::to_string(_i) + "_R2.png", lbp_vis);
    // _LBPCode_r3.convertTo(lbp_vis, CV_8U);  // 0-255 → 0-255 (уже в нужном диапазоне)
    // cv::imwrite(SRC_DIR"/../" + std::to_string(_i) + "_R3.png", lbp_vis);
}

cv::Mat LBPH::LBPCode(int radius) noexcept
{
    // Результат: uniform LBP (10 бинов: 0-8 единиц + 9 non-uniform)
    cv::Mat result = cv::Mat::zeros(_img.size(), CV_8UC1);

    const float angles[8] = {
        0.0f,
        -std::numbers::pi_v<float> / 4,
        -std::numbers::pi_v<float> / 2,
        -3 * std::numbers::pi_v<float> / 4,
        std::numbers::pi_v<float>,
        3 * std::numbers::pi_v<float> / 4,
        std::numbers::pi_v<float> / 2,
        std::numbers::pi_v<float> / 4
    };

    for (int row = radius; row < _img.rows - radius; ++row)
    {
        for (int col = radius; col < _img.cols - radius; ++col)
        {
            uchar center = _img.at<uchar>(row, col);
            uchar lbp_code = 0;

            for (int p = 0; p < 8; ++p)
            {
                float x = col + radius * std::cos(angles[p]);
                float y = row + radius * std::sin(angles[p]);

                int x0 = static_cast<int>(std::floor(x));
                int y0 = static_cast<int>(std::floor(y));
                int x1 = std::min(x0 + 1, _img.cols - 1);
                int y1 = std::min(y0 + 1, _img.rows - 1);
                x0 = std::max(x0, 0);
                y0 = std::max(y0, 0);

                float dx = x - x0;
                float dy = y - y0;

                float interpolated =
                    _img.at<uchar>(y0, x0) * (1-dx) * (1-dy) +
                    _img.at<uchar>(y0, x1) * dx     * (1-dy) +
                    _img.at<uchar>(y1, x0) * (1-dx) * dy     +
                    _img.at<uchar>(y1, x1) * dx     * dy;

                if (interpolated >= static_cast<float>(center))
                    lbp_code |= (1 << p);
            }

            result.at<uchar>(row, col) = ToUniformCode(lbp_code);
        }
    }

    return result;
}

cv::Mat LBPH::Histogram()
{
    // Гистограмма по сетке GRID_R x GRID_C блоков, конкатенация в один вектор
    // Каждый блок: 10 бинов × 3 радиуса = 30 значений
    // Итого: GRID_R * GRID_C * 30 строк, 1 столбец
    // Но сохраняем в 10 бинов × 3 столбца (как в прототипе — 256 строк)
    // Используем стандартные 10 бинов uniform LBP

    constexpr int BINS = 10; // uniform LBP бинов (0-8 единиц + non-uniform)
    constexpr int GRID_R = 8;
    constexpr int GRID_C = 8;
    constexpr int TOTAL_ROWS = GRID_R * GRID_C * BINS; // 640

    // Веса блоков: центр лица важнее краёв
    // Гауссово распределение весов по сетке
    // auto blockWeight = [&](int gr, int gc) -> double {
    //     double cy = (GRID_R - 1) / 2.0;
    //     double cx = (GRID_C - 1) / 2.0;
    //     double dy = (gr - cy) / cy;
    //     double dx = (gc - cx) / cx;
    //     return std::exp(-0.5 * (dx*dx + dy*dy) * 2.0);
    // };

    // Возвращаем матрицу TOTAL_ROWS × 3
    cv::Mat result = cv::Mat::zeros(TOTAL_ROWS, 3, CV_64FC1);

    auto fillGrid = [&](const cv::Mat& lbp, int radius, int col_idx)
    {
        int cell_h = (lbp.rows - 2 * radius) / GRID_R;
        int cell_w = (lbp.cols - 2 * radius) / GRID_C;

        for (int gr = 0; gr < GRID_R; ++gr)
        {
            for (int gc = 0; gc < GRID_C; ++gc)
            {
                int row_start = radius + gr * cell_h;
                int col_start = radius + gc * cell_w;
                int base_bin  = (gr * GRID_C + gc) * BINS;

                for (int r = row_start; r < row_start + cell_h; ++r)
                {
                    for (int c = col_start; c < col_start + cell_w; ++c)
                    {
                        uchar bin = lbp.at<uchar>(r, c); // 0..9
                        result.at<double>(base_bin + bin, col_idx) += 1.0;
                    }
                }
            }
        }
    };

    // auto fillGrid = [&](const cv::Mat& lbp, int radius, int col_idx)
    // {
    //     int cell_h = (lbp.rows - 2 * radius) / GRID_R;
    //     int cell_w = (lbp.cols - 2 * radius) / GRID_C;

    //     for (int gr = 0; gr < GRID_R; ++gr)
    //     {
    //         for (int gc = 0; gc < GRID_C; ++gc)
    //         {
    //             int row_start = radius + gr * cell_h;
    //             int col_start = radius + gc * cell_w;
    //             int base_bin  = (gr * GRID_C + gc) * BINS;
    //             double wt     = blockWeight(gr, gc);

    //             for (int r = row_start; r < row_start + cell_h; ++r)
    //                 for (int c = col_start; c < col_start + cell_w; ++c)
    //                 {
    //                     uchar bin = lbp.at<uchar>(r, c);
    //                     result.at<double>(base_bin + bin, col_idx) += wt;
    //                 }
    //         }
    //     }
    // };

    fillGrid(_LBPCode_r1, 1, 0);
    fillGrid(_LBPCode_r2, 2, 1);
    fillGrid(_LBPCode_r3, 3, 2);

    return result;
}

void LBPH::NormalizeHistogram()
{
    _normalizeHistogram = Histogram();

    for (int col = 0; col < 3; ++col)
    {
        double sum = cv::sum(_normalizeHistogram.col(col))[0];
        if (sum > 0.0)
            _normalizeHistogram.col(col) /= sum;
    }
}

double LBPH::Similarity(const LBPH& standart, const LBPH& tested)
{
    const double w[3]   = {0.5, 0.3, 0.2};
    const double alpha  = 0.6;
    const double beta   = 1 - alpha;

    double chi2_total   = 0.0;
    double cosine_total = 0.0;
    int    n_rows       = standart._normalizeHistogram.rows;

    for (int col = 0; col < 3; ++col)
    {
        double chi2_sum    = 0.0;
        double dot         = 0.0;
        double norm_A      = 0.0;
        double norm_B      = 0.0;

        for (int i = 0; i < n_rows; ++i)
        {
            double a = standart._normalizeHistogram.at<double>(i, col);
            double b = tested._normalizeHistogram.at<double>(i, col);

            dot   += a * b;
            norm_A += a * a;
            norm_B += b * b;

            double sum = a + b;
            if (sum > 1e-10)
                chi2_sum += (a - b) * (a - b) / sum;
        }

        double denom = std::sqrt(norm_A) * std::sqrt(norm_B);
        double cosine_sim = (denom > 1e-10) ? dot / denom : 0.0;
        double chi2_sim   = 1.0 / (1.0 + chi2_sum);

        chi2_total   += w[col] * chi2_sim;
        cosine_total += w[col] * cosine_sim;
    }

    // std::cout   << "Similarity="
    //             << alpha * chi2_total + beta * cosine_total << std::endl;
    // std::cout   << "\nW={"      << w[0] << ", "
    //                             << w[1] << ", "
    //                             << w[2] << "}"
    //             << std::endl;
    // std::cout   << "===============" << std::endl;

    return alpha * chi2_total + beta * cosine_total;
}
} // namespace LBPH::cpu