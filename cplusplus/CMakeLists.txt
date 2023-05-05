# 指定 CMake 的最低版本，请自行修改
cmake_minimum_required(VERSION 3.15)

project(rtdetr-onnxruntime VERSION 1.0 LANGUAGES CXX)

add_compile_options(-std=c++11)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# opencv
set(OpenCV_DIR "/usr/local" CACHE PATH "Location of opencv libraries" FORCE)

# onnxruntime 替换为自己的onnxruntime安装编译路径
set(ONNXRUNTIME_ROOT_PATH "path/to/your/custom/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)


if (NOT DEFINED OpenCV_DIR OR ${OpenCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OpenCV_DIR with -DOpenCV_DIR=/path/to/your/custom/opencv")
endif()

# 将生成的可执行文件保存到指定路径
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")

# onnxruntime 头文件
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# onnxruntime 貌似不支持 find_package，此处仅添加 OpenCV
find_package(OpenCV REQUIRED core imgcodecs imgproc highgui PATHS ${OpenCV_DIR})

# 编译成可执行文件
add_executable(main main.cpp)

target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBS} )


