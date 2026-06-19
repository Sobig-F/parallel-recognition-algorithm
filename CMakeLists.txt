cmake_minimum_required(VERSION 3.18)
project(BiometricCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures to build for")

set(OpenCV_DIR "D:/OpenCV/install/x64/vc17/lib" CACHE PATH "OpenCV config directory")
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    face
    cudaarithm
    cudaimgproc
    cudafilters
    cudawarping
    imgcodecs
    dnn
    highgui
    objdetect
)

find_package(CUDAToolkit REQUIRED)

set(ONNX_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data/models/face_detection_yunet_2023mar.onnx")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

add_executable(biometric
    src/main.cpp
    src/app/face_database.cpp
    src/app/face_database.hpp
    src/app/recognition_engine.cpp
    src/app/recognition_engine.hpp
    src/app/win_gui.cpp
    src/app/win_gui.hpp
    src/LBPH/lbph_descriptors.cpp
    src/LBPH/lbph_descriptors.hpp
    src/LBPH/lbph_descriptors_cuda.cu
    src/preproc/cropFace.cpp
    src/preproc/cropFace.hpp
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

target_include_directories(biometric PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(MSVC)
    target_compile_options(biometric PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

target_link_libraries(biometric
    ${OpenCV_LIBS}
    CUDA::cudart
    comctl32
    comdlg32
    shell32
    ole32
)

target_compile_definitions(biometric PRIVATE
    ONNX_MODEL_PATH="${ONNX_MODEL_PATH}"
    SRC_DIR="${SRC_DIR}"
)
