#include <filesystem>

#include "LBPH/CPU/lbph_cpu.hpp"
#include "preproc/cropFace.hpp"
#include "preproc/preproc.hpp"

// #define EPS 0.55

int main(int argc, char* argv[])
{
    double eps = 0.9;
    if (argc > 1)
    {
        char* end;
        eps = strtod(argv[1], &end);
    }
    

    cv::Mat img1 = cv::imread(SRC_DIR"../data/Faces/Akshay Kumar_24.jpg",
                            cv::IMREAD_COLOR
                        );
    
    preprocessing::FaceDetector detector(ONNX_MODEL_PATH);

    cv::Mat standart = detector.Detect(img1);
    preprocessing::GrayScale(standart);
    preprocessing::Resize(standart);
    int i = 0;
    for (const auto& entry : std::filesystem::directory_iterator(SRC_DIR"../data/Faces"))
    {
        if (std::filesystem::is_regular_file(entry.path()))
        {
            ++i;
            cv::Mat test_img_input = cv::imread(entry.path().string(),
                                                cv::IMREAD_COLOR
                                        );
            cv::Mat test_img = detector.Detect(test_img_input);
            if (test_img.empty())
            {
                continue;
            }
            preprocessing::GrayScale(test_img);
            preprocessing::Resize(test_img);
            LBPH::cpu::LBPH tested(test_img);
            double similarity = LBPH::cpu::LBPH::Similarity(standart, tested);
            if (similarity >= eps)
            {
                cv::imwrite(SRC_DIR"../result/true/" + std::to_string(i) + "_" + std::to_string(similarity) + ".jpg", test_img);
            } else
            {
                cv::imwrite(SRC_DIR"../result/false/" + std::to_string(i) + "_" + std::to_string(similarity) + ".jpg", test_img);
            }
            std::cout << i << std::endl;
        }
    }
}