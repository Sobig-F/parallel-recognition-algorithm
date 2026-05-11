cmake_minimum_required(VERSION 3.18)
project(BiometricCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 89)

# ==================== OPENCV ====================
set(OpenCV_DIR "D:/OpenCV/install/x64/vc17/lib")
find_package(OpenCV REQUIRED COMPONENTS 
    core imgproc face cudaarithm cudaimgproc cudafilters cudawarping
    imgcodecs dnn highgui objdetect)

# ==================== CUDA TOOLKIT ====================
find_package(CUDAToolkit REQUIRED)

# ==================== EXECUTABLE ====================
file(GLOB_RECURSE PREPROC 
    "src/preproc/*"
)
file(GLOB_RECURSE APP
    "src/app/*"
)
file(GLOB_RECURSE ARCFACE 
    "src/arcface/*"
)
file(GLOB_RECURSE FISHERFACES 
    "src/fisherfaces/*"
)
file(GLOB_RECURSE LBPH 
    "src/LBPH/*"
)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS 
             "src/preproc/*"
             "src/app/*"
             "src/arcface/*"
             "src/fisherfaces/*"
             "src/LBPH/*"
)

add_executable(biometric
    src/main.cpp
    ${PREPROC}
    ${APP}
    ${ARCFACE}
    ${FISHERFACES}
    ${LBPH}
    # src/cores/test.cu  # Файл с вашими ядрами
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "")  # Убрать -G из дебаг сборки
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

set(ONNX_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data/models/face_detection_yunet_2023mar.onnx")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

target_include_directories(biometric PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}  # Пути к заголовкам CUDA
)

if(MSVC)
    target_compile_options(biometric PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

target_link_libraries(biometric 
    ${OpenCV_LIBS}
    CUDA::cudart  # CUDA runtime из CUDAToolkit
    comctl32
    comdlg32
    shell32
    ole32
)

target_compile_definitions(biometric PRIVATE 
    ONNX_MODEL_PATH="${ONNX_MODEL_PATH}"
    SRC_DIR="${SRC_DIR}"
)
