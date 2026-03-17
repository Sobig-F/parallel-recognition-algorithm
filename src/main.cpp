#include <filesystem>

#include "LBPH/CPU/lbph_cpu.hpp"

// #define EPS 0.55

int main(int argc, char* argv[])
{
    char* end;
    double eps = strtod(argv[1], &end);
    
    std::unique_ptr<cv::Mat> img = std::make_unique<cv::Mat>(
                                        cv::imread("../../data/Faces/Robert Downey Jr_30.jpg",
                                            cv::IMREAD_COLOR
                                        )
                                    );
                                   
    LBPH::cpu::LBPH standart(move(img));
    for (const auto& entry : std::filesystem::directory_iterator("../../data/Faces"))
    {
        if (std::filesystem::is_regular_file(entry.path()))
        {
            std::unique_ptr<cv::Mat> test_img = std::make_unique<cv::Mat>(
                                        cv::imread(entry.path().string(),
                                                cv::IMREAD_COLOR
                                        )
                                    );
            LBPH::cpu::LBPH tested(move(test_img));
            if (LBPH::cpu::LBPH::Distance(standart, tested) >= eps)
            {
                std::filesystem::copy_file(entry.path().string(), "../../result/" + entry.path().filename().string());
            }
        }
    }
}