#include <filesystem>

#include "LBPH/CPU/lbph_cpu.hpp"
#include "preproc/cropFace.hpp"

// #define EPS 0.55

int main(int argc, char* argv[])
{
    char* end;
    double eps = strtod(argv[1], &end);

    cv::Mat img1 = cv::imread(SRC_DIR"../data/photo_260@28-01-2022_16-39-21.jpg",
                            cv::IMREAD_COLOR
                        );
    cv::Mat img2 = cv::imread(SRC_DIR"../data/photo/photo_151@28-01-2022_16-33-41.jpg",
                            cv::IMREAD_COLOR
                        );
    
    preprocessing::FaceDetector detector(ONNX_MODEL_PATH);
    
    cv::Mat result1 = detector.Detect(img1);
    preprocessing::GrayScale(result1);
    preprocessing::GaussainBlur(result1);
    preprocessing::CLAHE(result1);
    preprocessing::Resize(result1);

    cv::Mat result2 = detector.Detect(img2);
    preprocessing::GrayScale(result2);
    preprocessing::GaussainBlur(result2);
    preprocessing::CLAHE(result2);
    preprocessing::Resize(result2);

    LBPH::cpu::LBPH test1(result1);
    LBPH::cpu::LBPH test2(result2);

    std::cout << LBPH::cpu::LBPH::Similarity(test2, test1) << std::endl;
    // int i = 0;
    // for (const auto& entry : std::filesystem::directory_iterator(SRC_DIR"../data/photo"))
    // {
    //     if (std::filesystem::is_regular_file(entry.path()))
    //     {
    //         ++i;
    //         cv::Mat test_img_input = cv::imread(entry.path().string(),
    //                                             cv::IMREAD_COLOR
    //                                     );
    //         cv::Mat test_img = detector.Detect(test_img_input);
    //         if (test_img.empty())
    //         {
    //             continue;
    //         }
    //         preprocessing::GrayScale(test_img);
    //         preprocessing::GaussainBlur(test_img);
    //         preprocessing::CLAHE(test_img);
    //         preprocessing::Resize(test_img);
    //         LBPH::cpu::LBPH tested(test_img);
    //         if (LBPH::cpu::LBPH::Similarity(standart, tested) >= eps)
    //         {
    //             // std::cout << entry.path().filename().string() << std::endl;
    //             cv::imwrite(SRC_DIR"../result/" + entry.path().filename().string(), test_img);
    //         }
    //         std::cout << i << std::endl;
    //     }
    // }
}