cmake_minimum_required(VERSION 3.10)
project(MyOpenCVProject)

# Путь к установленному OpenCV
set(OpenCV_DIR "D:/OpenCV/install/x64/vc17/lib")
find_package(OpenCV REQUIRED COMPONENTS 
    core imgproc imgcodecs videoio highgui
    dnn cudaarithm cudaimgproc cudafilters cudacodec
    features2d xfeatures2d tracking quality face
    optflow stitching video cudastereo cudaoptflow cudabgsegm aruco)

add_executable(lib_test lib_test.cpp)
target_link_libraries(lib_test ${OpenCV_LIBS})

target_compile_definitions(lib_test PRIVATE
    HAVE_CUDA
    HAVE_CUDNN
    HAVE_OPENCV_CUDACODEC
    HAVE_OPENCL
    HAVE_VULKAN)

# Опционально: копировать DLL при сборке
add_custom_command(TARGET lib_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "D:/OpenCV/build/install/x64/vc17/bin"
    $<TARGET_FILE_DIR:lib_test>)