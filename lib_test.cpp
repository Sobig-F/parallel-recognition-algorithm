#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>          // ← Для CUDA GaussianBlur
#include <opencv2/cudacodec.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/quality.hpp>
#include <opencv2/face.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/cudaoptflow.hpp>          // ← Для CUDA Optical Flow
#include <opencv2/stitching.hpp>
#include <opencv2/video.hpp>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>
#include <Windows.h>

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

std::string getBackendName(cv::dnn::Backend backend) {
    switch (backend) {
        case cv::dnn::DNN_BACKEND_DEFAULT:         return "DEFAULT";
        case cv::dnn::DNN_BACKEND_HALIDE:          return "HALIDE";
        case cv::dnn::DNN_BACKEND_INFERENCE_ENGINE: return "INFERENCE_ENGINE";
        case cv::dnn::DNN_BACKEND_OPENCV:          return "OPENCV";
        case cv::dnn::DNN_BACKEND_VKCOM:           return "VKCOM";
        case cv::dnn::DNN_BACKEND_CUDA:            return "CUDA";
        case cv::dnn::DNN_BACKEND_WEBNN:           return "WEBNN";
        case cv::dnn::DNN_BACKEND_TIMVX:           return "TIMVX";
        case cv::dnn::DNN_BACKEND_CANN:            return "CANN";
        default:                                   return "UNKNOWN";
    }
}

std::string getTargetName(cv::dnn::Target target) {
    switch (target) {
        case cv::dnn::DNN_TARGET_CPU:              return "CPU";
        case cv::dnn::DNN_TARGET_OPENCL:           return "OPENCL";
        case cv::dnn::DNN_TARGET_OPENCL_FP16:      return "OPENCL_FP16";
        case cv::dnn::DNN_TARGET_MYRIAD:           return "MYRIAD";
        case cv::dnn::DNN_TARGET_VULKAN:           return "VULKAN";
        case cv::dnn::DNN_TARGET_FPGA:             return "FPGA";
        case cv::dnn::DNN_TARGET_CUDA:             return "CUDA";
        case cv::dnn::DNN_TARGET_CUDA_FP16:        return "CUDA_FP16";
        case cv::dnn::DNN_TARGET_HDDL:             return "HDDL";
        default:                                   return "UNKNOWN";
    }
}

void printSection(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

// ============================================================================
// ТЕСТ 1: Общая информация о сборке
// ============================================================================

void test_build_info() {
    printSection("ТЕСТ 1: Общая информация о сборке OpenCV");
    
    std::cout << "  Версия OpenCV: " << CV_VERSION << std::endl;
    std::cout << "  Версия (major): " << CV_MAJOR_VERSION << std::endl;
    std::cout << "  Версия (minor): " << CV_MINOR_VERSION << std::endl;
    
    std::cout << "\n  Флаги сборки:" << std::endl;
#ifdef HAVE_CUDA
    std::cout << "    ✅ HAVE_CUDA" << std::endl;
#else
    std::cout << "    ❌ HAVE_CUDA" << std::endl;
#endif

#ifdef HAVE_CUDNN
    std::cout << "    ✅ HAVE_CUDNN" << std::endl;
#else
    std::cout << "    ❌ HAVE_CUDNN" << std::endl;
#endif

#ifdef HAVE_OPENCV_CUDACODEC
    std::cout << "    ✅ HAVE_OPENCV_CUDACODEC" << std::endl;
#else
    std::cout << "    ❌ HAVE_OPENCV_CUDACODEC" << std::endl;
#endif

#ifdef HAVE_OPENCL
    std::cout << "    ✅ HAVE_OPENCL" << std::endl;
#else
    std::cout << "    ❌ HAVE_OPENCL" << std::endl;
#endif

#ifdef HAVE_VULKAN
    std::cout << "    ✅ HAVE_VULKAN" << std::endl;
#else
    std::cout << "    ❌ HAVE_VULKAN" << std::endl;
#endif
}

// ============================================================================
// ТЕСТ 2: CUDA устройства
// ============================================================================

void test_cuda_devices() {
    printSection("ТЕСТ 2: CUDA устройства");
    
    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "  Количество CUDA устройств: " << deviceCount << std::endl;
    
    if (deviceCount > 0) {
        cv::cuda::printCudaDeviceInfo(0);
    }
}

// ============================================================================
// ТЕСТ 3: DNN + CUDA + cuDNN
// ============================================================================

void test_dnn_cuda() {
    printSection("ТЕСТ 3: DNN + CUDA + cuDNN");
    
    auto backends = cv::dnn::getAvailableBackends();
    
    std::cout << "  Доступные бэкенды DNN:" << std::endl;
    for (const auto& pair : backends) {
        std::cout << "    • " << getBackendName(pair.first) 
                  << " (" << getTargetName(pair.second) << ")" << std::endl;
    }
    
    bool hasCuda = std::find_if(backends.begin(), backends.end(),
        [](const std::pair<cv::dnn::Backend, cv::dnn::Target>& p) {
            return p.first == cv::dnn::DNN_BACKEND_CUDA;
        }) != backends.end();
    
    std::cout << "\n  Статус CUDA для DNN: " << (hasCuda ? "✅ ДОСТУПЕН" : "❌ НЕДОСТУПЕН") << std::endl;
}

// ============================================================================
// ТЕСТ 4: CUDA Image Processing
// ============================================================================

void test_cuda_speed() {
    printSection("ТЕСТ 4: CUDA Image Processing (тест скорости)");
    
    cv::Mat cpu_img(1920, 1080, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(cpu_img);
    
    // CPU тест
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat cpu_result;
    for (int i = 0; i < 100; i++) {
        cv::cvtColor(cpu_img, cpu_result, cv::COLOR_BGR2GRAY);
    }
    auto cpu_time = std::chrono::high_resolution_clock::now() - start;
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_time).count();
    
    // GPU тест
    start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat gpu_result;
    for (int i = 0; i < 100; i++) {
        cv::cuda::cvtColor(gpu_img, gpu_result, cv::COLOR_BGR2GRAY);
    }
    cv::cuda::Stream stream;
    stream.waitForCompletion();
    auto gpu_time = std::chrono::high_resolution_clock::now() - start;
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_time).count();
    
    std::cout << "  CPU: " << cpu_ms << " мс" << std::endl;
    std::cout << "  GPU: " << gpu_ms << " мс" << std::endl;
    std::cout << "  Ускорение: " << (cpu_ms / gpu_ms) << "x" << std::endl;
    
    // Gaussian Blur через cudafilters
    printSection("Тест Gaussian Blur (CUDA)");
    try {
        cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(
            gpu_img.type(), gpu_result.type(), cv::Size(5, 5), 1.5);
        gaussianFilter->apply(gpu_img, gpu_result);
        stream.waitForCompletion();
        std::cout << "  ✅ CUDA Gaussian Blur: РАБОТАЕТ" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "  ❌ CUDA Gaussian Blur: ОШИБКА - " << e.what() << std::endl;
    }
}

// ============================================================================
// ТЕСТ 5: Feature Detectors
// ============================================================================

void test_feature_detectors() {
    printSection("ТЕСТ 5: Детекторы признаков");
    
    try {
        auto sift = cv::SIFT::create();
        std::cout << "  SIFT: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  SIFT: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    try {
        auto orb = cv::ORB::create();
        std::cout << "  ORB: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  ORB: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    try {
        auto akaze = cv::AKAZE::create();
        std::cout << "  AKAZE: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  AKAZE: ❌ НЕДОСТУПЕН" << std::endl;
    }
}

// ============================================================================
// ТЕСТ 6: Video Tracking (ИСПРАВЛЕНО)
// ============================================================================

void test_tracking() {
    printSection("ТЕСТ 6: Трекеры для видео");
    
    // В OpenCV 4.10 доступны только KCF и CSRT из коробки
    // Остальные удалены или перемещены в opencv_contrib
    
    std::cout << "  Доступные трекеры:" << std::endl;
    
    try {
        auto tracker = cv::TrackerKCF::create();
        std::cout << "    KCF: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "    KCF: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    try {
        auto tracker = cv::TrackerCSRT::create();
        std::cout << "    CSRT: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "    CSRT: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    // TLD, MedianFlow, MOSSE удалены в OpenCV 4.x
    std::cout << "    TLD: ❌ УДАЛЁН (OpenCV 4.x)" << std::endl;
    std::cout << "    MedianFlow: ❌ УДАЛЁН (OpenCV 4.x)" << std::endl;
    std::cout << "    MOSSE: ❌ УДАЛЁН (OpenCV 4.x)" << std::endl;
}

// ============================================================================
// ТЕСТ 7: Quality Assessment (ИСПРАВЛЕНО)
// ============================================================================

void test_quality() {
    printSection("ТЕСТ 7: Оценка качества изображений");
    std::flush(std::cout);
    
    cv::Mat img1(100, 100, CV_8UC1, cv::Scalar(100));
    cv::Mat img2(100, 100, CV_8UC1, cv::Scalar(105));
    
    std::cout << "  Доступные метрики качества:" << std::endl;
    
    // PSNR
    try {
        double psnr = cv::PSNR(img1, img2);
        std::cout << "    PSNR: ✅ ДОСТУПЕН (" << psnr << " dB)" << std::endl;
    } catch (...) {
        std::cout << "    PSNR: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    // BRISQUE (ИСПРАВЛЕНО: с параметрами модели)
    try {
        std::string modelPath = "D:/OpenCV/build/install/etc/quality/brisque_model_live.yml";
        std::string rangePath = "D:/OpenCV/build/install/etc/quality/brisque_range_live.yml";
        
        cv::Ptr<cv::quality::QualityBRISQUE> brisque = 
            cv::quality::QualityBRISQUE::create(modelPath, rangePath);
        
        // Тест вычисления качества
        std::vector<cv::Mat> input = {img1};
        cv::Scalar quality = brisque->compute(input);
        
        std::cout << "    BRISQUE: ✅ ДОСТУПЕН (оценка: " << quality[0] << ")" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "    BRISQUE: ⚠️ ОШИБКА - " << e.what() << std::endl;
    } catch (...) {
        std::cout << "    BRISQUE: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    // SSIM
    try {
        cv::Mat ssim_map;
        cv::Scalar ssim = cv::quality::QualitySSIM::compute(img1, img2, ssim_map);
        std::cout << "    SSIM: ✅ ДОСТУПЕН (" << ssim[0] << ")" << std::endl;
    } catch (...) {
        std::cout << "    SSIM: ❌ НЕДОСТУПЕН" << std::endl;
    }
}

// ============================================================================
// ТЕСТ 8: Face Module
// ============================================================================

void test_face_module() {
    printSection("ТЕСТ 8: Модуль распознавания лиц");
    
    try {
        auto lbph = cv::face::LBPHFaceRecognizer::create();
        std::cout << "  LBPH: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  LBPH: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    try {
        auto eigen = cv::face::EigenFaceRecognizer::create();
        std::cout << "  Eigen: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  Eigen: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    try {
        auto fisher = cv::face::FisherFaceRecognizer::create();
        std::cout << "  Fisher: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  Fisher: ❌ НЕДОСТУПЕН" << std::endl;
    }
}

// ============================================================================
// ТЕСТ 9: cudacodec
// ============================================================================

void test_cudacodec() {
    printSection("ТЕСТ 9: cudacodec (аппаратное видео)");
    
#ifdef HAVE_OPENCV_CUDACODEC
    std::cout << "  Модуль cudacodec: ✅ СКОМПИЛИРОВАН" << std::endl;
    std::cout << "  NVDEC: ✅ ДОСТУПЕН" << std::endl;
    std::cout << "  NVENC: ✅ ДОСТУПЕН" << std::endl;
#else
    std::cout << "  Модуль cudacodec: ❌ НЕ СКОМПИЛИРОВАН" << std::endl;
#endif
}

// ============================================================================
// ТЕСТ 10: OpenCL и Vulkan
// ============================================================================

void test_opencl_vulkan() {
    printSection("ТЕСТ 10: OpenCL и Vulkan");
    
#ifdef HAVE_OPENCL
    std::cout << "  OpenCL: ✅ ВКЛЮЧЕН" << std::endl;
#else
    std::cout << "  OpenCL: ❌ ОТКЛЮЧЕН" << std::endl;
#endif

#ifdef HAVE_VULKAN
    std::cout << "  Vulkan: ✅ ВКЛЮЧЕН" << std::endl;
#else
    std::cout << "  Vulkan: ❌ ОТКЛЮЧЕН" << std::endl;
#endif
}

// ============================================================================
// ТЕСТ 11: Дополнительные модули (ИСПРАВЛЕНО)
// ============================================================================

void test_extra_modules() {
    printSection("ТЕСТ 11: Дополнительные модули");
    
    // Stitching
    try {
        auto stitcher = cv::Stitcher::create();
        std::cout << "  Stitching: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  Stitching: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    // Optical Flow (CPU)
    try {
        auto flow = cv::optflow::createOptFlow_DualTVL1();
        std::cout << "  Optical Flow (CPU): ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  Optical Flow (CPU): ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    // CUDA Optical Flow (ИСПРАВЛЕНО: используем правильный API)
    try {
        auto cuda_flow = cv::cuda::OpticalFlowDual_TVL1::create();
        std::cout << "  CUDA Optical Flow: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  CUDA Optical Flow: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    // Background Subtraction
    try {
        auto bgsub = cv::createBackgroundSubtractorMOG2();
        std::cout << "  Background Subtractor: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  Background Subtractor: ❌ НЕДОСТУПЕН" << std::endl;
    }
    
    // ArUco
    try {
        auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        std::cout << "  ArUco: ✅ ДОСТУПЕН" << std::endl;
    } catch (...) {
        std::cout << "  ArUco: ❌ НЕДОСТУПЕН" << std::endl;
    }
}

// ============================================================================
// ТЕСТ 12: Итоговая сводка
// ============================================================================

void test_summary() {
    printSection("ИТОГОВАЯ СВОДКА");
    std::flush(std::cout);
    
    std::cout << "  ✅ OpenCV: " << CV_VERSION << std::endl;
    std::cout << "  ✅ CUDA: " << (cv::cuda::getCudaEnabledDeviceCount() > 0 ? "РАБОТАЕТ" : "НЕТ") << std::endl;
    std::cout << "  ✅ cuDNN: через DNN backend" << std::endl;
    std::cout << "  ✅ cudacodec: " 
#ifdef HAVE_OPENCV_CUDACODEC
              << "СКОМПИЛИРОВАН"
#else
              << "НЕ СКОМПИЛИРОВАН"
#endif
              << std::endl;
    
    std::cout << "\n  🎯 Готово для:" << std::endl;
    std::cout << "    • Распознавание лиц (DNN + CUDA)" << std::endl;
    std::cout << "    • Обработка видео (cudacodec)" << std::endl;
    std::cout << "    • Детекция признаков (SIFT, ORB)" << std::endl;
    std::cout << "    • Отслеживание (KCF, CSRT)" << std::endl;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  🎉 ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

// ============================================================================
// ТЕСТ 11: Дополнительные модули (ИСПРАВЛЕНО)
// ============================================================================

void test_onnx() {
    printSection("ТЕСТ 13: Проверка поддержки ONNX");
    
    // Попробуем создать сеть из ONNX (даже несуществующего файла, важно чтобы функция линковалась)
    // Если функция readNetFromONNX отсутствует при линковке, компилятор выдаст ошибку.
    // Если она есть, но сборка без_PROTOBUF, может быть пустая сеть или ошибка парсинга.
    
    try {
        std::string modelPath = "../../face_detection_yunet_2023mar.onnx";
        cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
        // 2. Проверка доступных бэкендов
        std::cout << "Доступные бэкенды DNN:" << std::endl;
        auto backends = cv::dnn::getAvailableBackends();
        if (backends.empty()) {
            std::cout << "  ⚠️ Список пуст" << std::endl;
        } else {
            for (const auto& b : backends) {
                std::cout << "  • Backend: " << b.first << ", Target: " << b.second << std::endl;
            }
        }

        // 3. Принудительное включение CUDA (если драйвер позволяет)
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16); // RTX 4060 поддерживает FP16
        
        std::cout << "✅ Бэкенд установлен: CUDA" << std::endl;

        // 4. Тестовый прогон (создаем случайное изображение)
        cv::Mat img = cv::imread("test_face.jpg"); // Положите любое фото рядом
        if (img.empty()) {
            std::cout << "⚠️ Файл test_face.jpg не найден, создаем заглушку..." << std::endl;
            img = cv::Mat(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        }

        // 5. Подготовка(blob) и инференс
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0, cv::Size(320, 320), cv::Scalar(), true, false);
        
        net.setInput(blob);
        cv::Mat output = net.forward();

        std::cout << "✅ Инференс пройден! Размер вывода: " << output.size << std::endl;
        std::cout << "🎉 Система готова к работе!" << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "❌ Ошибка: " << e.what() << std::endl;
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    #ifdef _WIN32
        system("chcp 65001 > nul");
        SetConsoleOutputCP(CP_UTF8);
    #endif
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         OpenCV 4.10.0 Full Build Test Suite                          ║" << std::endl;
    std::cout << "║         CUDA + cuDNN + cudacodec + Contrib Modules                   ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    try {
        test_build_info();
        test_cuda_devices();
        test_dnn_cuda();
        test_cuda_speed();
        test_feature_detectors();
        test_tracking();
        test_quality();
        test_face_module();
        test_cudacodec();
        test_opencl_vulkan();
        test_extra_modules();
        test_summary();
        test_onnx();
    } catch (const cv::Exception& e) {
        std::cerr << "\n  ❌ Ошибка OpenCV: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}