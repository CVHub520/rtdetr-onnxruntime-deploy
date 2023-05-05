// Author: CVHub
// Date: May 2023
// Description: The project demonstrates how to deploy the RT-DETR model in C++ using ONNXRUNTIME.

#include <unistd.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>


using namespace cv;
using namespace std;
using namespace Ort;


std::vector<std::string> readLabels(const std::string& labelPath) {
    std::vector<std::string> labels;
    std::ifstream infile(labelPath);
    std::string line;
    while (std::getline(infile, line)) {
        labels.push_back(line);
    }
    infile.close();
    return labels;
}


size_t vectorProduct(const std::vector<int64_t>& vector) {
    if (vector.empty())
        return 0;
    
    size_t product = 1;
    for (const auto& element : vector)
        product *= element;
    
    return product;
}

vector<vector<float>> bbox_cxcywh_to_xyxy(const vector<vector<float>>& boxes)
{
    vector<vector<float>> xyxy_boxes;
    for (const auto& box : boxes)
    {
        float x1 = box[0] - box[2] / 2.0f;
        float y1 = box[1] - box[3] / 2.0f;
        float x2 = box[0] + box[2] / 2.0f;
        float y2 = box[1] + box[3] / 2.0f;
        xyxy_boxes.push_back({ x1, y1, x2, y2 });
    }
    return xyxy_boxes;
}


bool is_normalized(const std::vector<std::vector<float>>& values) {
    for (const auto& row : values) {
        for (const auto& val : row) {
            if (val <= 0 || val >= 1) {
                return false;
            }
        }
    }
    return true;
}

void normalize_scores(std::vector<std::vector<float>>& scores) {
    for (auto& row : scores) {
        for (auto& val : row) {
            val = 1 / (1 + std::exp(-val));
        }
    }
}

vector<vector<int>> generate_class_colors(int num_classes) {
    vector<vector<int>> class_colors(num_classes, vector<int>(3));
    for (int i = 0; i < num_classes; ++i) {
        class_colors[i][0] = rand() % 256;
        class_colors[i][1] = rand() % 256;
        class_colors[i][2] = rand() % 256;
    }
    return class_colors;
}

void draw_boxes_and_save_image(
    const std::vector<int>& labels, 
    const std::vector<float>& scores, 
    const std::vector<std::vector<float>>& boxes, 
    const std::string& save_path, 
    const std::vector<std::string>& CLASS_NAMES,
    cv::Mat& im0
) {
    vector<vector<int>> CLASS_COLORS = generate_class_colors(CLASS_NAMES.size());

    for (size_t i = 0; i < boxes.size(); ++i) {
        int label = labels[i];
        float score = scores[i];
        std::ostringstream oss;
        oss << CLASS_NAMES[label] << ": " << std::fixed << std::setprecision(2) << score;
        std::string label_text = oss.str();
        cv::Rect rect((int)boxes[i][0], (int)boxes[i][1], (int)(boxes[i][2] - boxes[i][0]), (int)(boxes[i][3] - boxes[i][1]));
        cv::Scalar color(CLASS_COLORS[label][0], CLASS_COLORS[label][1], CLASS_COLORS[label][2]);
        cv::rectangle(im0, rect, color, 2);
        cv::putText(im0, label_text, cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    cv::imwrite(save_path, im0);
}

int main(int argc, char* argv[])
{

    // 检查当前环境工作路径
    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("[INFO] Current working directory is: %s\n", cwd);
    } else {
        perror("getcwd() error");
        return 1;
    }

    // Note: There is not a C++ API that returns ORT version. 
    // Only C, so you shold include <onnxruntime_c_api.h>
    std::cout << "[INFO] ONNXRuntime version: " << OrtGetApiBase()->GetVersionString() << std::endl;
    
    bool useCUDA = false;
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";

    if (argc == 1) {
        useCUDA = false;
    }
    else if ( (argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0) ) {
        useCUDA = true;
    }
    else if ( (argc == 2) && (strcmp(argv[1], useCPUFlag) == 0) ) {
        useCUDA = false;
    }
    else {
        throw std::runtime_error("Invalid #Param, please check double again!");
    }

    if (useCUDA) {
        std::cout << "[INFO] Inference execution provider: CUDA" << std::endl;
    }
    else {
        std::cout << "[INFO] Inference execution provider: CPU" << std::endl;
    }

    std::string imagePath = "../../images/bus.jpg";
    std::string savePath = "../../images/bus_cpp_result.jpg";
    std::string labelPath = "../labels.txt";
    std::string mdoelPath = "../../weights/rtdetr_r50vd_6x_coco_cvhub.onnx";
    std::string instanceName = "rtdetr-onnxruntime-inference";
    size_t deviceId = 0;
    size_t batchSize = 1;
    float confThreshold = 0.45;
    
    std::vector<std::string> labels = readLabels(labelPath);
    
    if (labels.empty()) {
        throw std::runtime_error("No labels found!");
    }

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = deviceId;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Sets graph optimization level [Available levels are as below]
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals) 
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    // Create session
    Ort::Session ortSession(env, mdoelPath.c_str(), sessionOptions);

    // 创建一个以默认选项为基础的分配器对象，以便为后续的操作提供内存分配功能
    Ort::AllocatorWithDefaultOptions allocator;

    // 获取输入节点数量
    size_t numInputNodes = ortSession.GetInputCount();
    // 获取输出节点数量
    size_t numOutputNodes = ortSession.GetOutputCount();

    // 获取输入节点名称和维度
    std::vector <std::string> inputNodeNames;
    std::vector <vector <int64_t>> inputNodeDims;
    for (int i = 0; i < numInputNodes; i++) {
        auto inputName = ortSession.GetInputNameAllocated(i, allocator);
        inputNodeNames.push_back(inputName.get());
        Ort::TypeInfo inputTypeInfo = ortSession.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputDims = inputTensorInfo.GetShape();

        // 返回值类型请参考：https://onnxruntime.ai/docs/api/c/group___global.html#gaec63cdda46c29b8183997f38930ce38e
        // 此处返回 1 代表第二种类型(从0开始计数)，因此这里是一个 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT 类型，即 32 位浮点数类型
        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

        if (inputDims.at(0) == -1)
        {
            std::cout << "[Warning] Got dynamic batch size. Setting output batch size to "
                    << batchSize << "." << std::endl;
            inputDims.at(0) = batchSize;
        }

        inputNodeDims.push_back(inputDims);

        std::cout << "[INFO] Input name and shape is: " << inputName.get() << " [";
        for (size_t j = 0; j < inputDims.size(); j++) {
            std::cout << inputDims[j];
            if (j != inputDims.size()-1) {
                std::cout << ",";
            }
        }
        std::cout << ']' << std::endl;
    }

    // 获取输出节点名称
    std::vector <std::string> outputNodeNames;
    std::vector <vector <int64_t>> outputNodeDims;
    for (int i = 0; i < numOutputNodes; i++) {
        auto outputName = ortSession.GetOutputNameAllocated(i, allocator);
        outputNodeNames.push_back(outputName.get());
        Ort::TypeInfo outputTypeInfo = ortSession.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputDims = outputTensorInfo.GetShape();

        if (outputDims.at(0) == -1)
        {
            std::cout << "[Warning] Got dynamic batch size. Setting output batch size to "
                    << batchSize << "." << std::endl;
            outputDims.at(0) = batchSize;
        }

        outputNodeDims.push_back(outputDims);

        std::cout << "[INFO] Output name and shape is: " << outputName.get() << " [";
        for (size_t j = 0; j < outputDims.size(); j++) {
            std::cout << outputDims[j];
            if (j != outputDims.size()-1) {
                std::cout << ",";
            }
        }
        std::cout << ']' << std::endl;
    }
    std::cout << "[INFO] Model was initialized." << std::endl;

    /*       Preprocess     */

    // 图像预处理
    cv::Mat imageBGR = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);

    // 源图像尺寸
    int64_t imageHeight = imageBGR.rows;
    int64_t imageWidth = imageBGR.cols;
    std::cout << "[INFO] Source image size (h, w) is [" << imageHeight << ", " << imageWidth << "]" << std::endl;

    // 模型输入尺寸
    int64_t inputHeight = inputNodeDims[0].at(2);
    int64_t inputWidth = inputNodeDims[0].at(3);

    // 缩放因子
    float ratioHeight = static_cast<float>(inputHeight) / imageHeight;
    float ratioWidth = static_cast<float>(inputWidth) / imageWidth;

    cv::Mat resizedImageBGR, resizedImageRGB, resizedImageNormRGB, resizedImageNormRGBCHW, preprocessedImage;

    // 图像缩放
    cv::resize(imageBGR, resizedImageBGR, 
            cv::Size(0, 0), ratioWidth, ratioHeight, cv::INTER_LINEAR);
    std::cout << "[INFO] [Preprocess] Resize" << std::endl;

    // 色域转换 [BGR -> RGB]
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    std::cout << "[INFO] [Preprocess] BGR to RGB" << std::endl;

    // 图像归一化
    resizedImageRGB.convertTo(resizedImageNormRGB, CV_32FC3, 1.0 / 255);
    std::cout << "[INFO] [Preprocess] Normalization" << std::endl;

    // 通道转换 [HWC -> CHW]
    float *blob = nullptr;
    blob = new float[resizedImageNormRGB.cols * resizedImageNormRGB.rows * resizedImageNormRGB.channels()];
    cv::Size floatImageSize {resizedImageNormRGB.cols, resizedImageNormRGB.rows};
    std::vector<cv::Mat> chw(resizedImageNormRGB.channels());
    for (int i = 0; i < resizedImageNormRGB.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(resizedImageNormRGB, chw);
    std::cout << "[INFO] [Preprocess] HWC to CHW" << std::endl;

    // 维度扩展 [CHW -> NCHW]
    std::vector<int64_t> inputTensorShape = {1, 3, inputHeight, inputWidth};

    size_t inputTensorSize = vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));
    std::cout << "[INFO] [Preprocess] CHW to NCHW" << std::endl;

    // 检查输入和输出节点名称是否为空
    for (const auto& inputNodeName : inputNodeNames) {
        if (std::string(inputNodeName).empty()) {
            std::cerr << "Empty input node name found." << std::endl;
            return 1;
        }
    }

    // 格式转换
    std::vector<const char*> inputNodeNamesCStr;
    for (const auto& inputName : inputNodeNames) {
        inputNodeNamesCStr.push_back(inputName.c_str());
    }
    std::vector<const char*> outputNodeNamesCStr;
    for (const auto& outputName : outputNodeNames) {
        outputNodeNamesCStr.push_back(outputName.c_str());
    }

    /*     Inference       */
    std::vector<Ort::Value> outputTensors = ortSession.Run(
        Ort::RunOptions{nullptr}, 
        inputNodeNamesCStr.data(),
        inputTensors.data(), 
        inputTensors.size(),
        outputNodeNamesCStr.data(),
        1
    );
    std::cout << "[INFO] [Inference] Successfully!" << std::endl;

    /*   Post-Preprocess  */

    // 获取输出结果
    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);
    std::cout << "[INFO] [Postprocess] Get output results" << std::endl;

    // 提取 boxes 和 scores 分数
    int num_boxes = outputShape[1];
    int num_classes = labels.size();
    vector<vector<float>> boxes(num_boxes, vector<float>(4));
    vector<vector<float>> scores(num_boxes, vector<float>(num_classes));
    int score_start_index = 4;
    int score_end_index = 4 + num_classes;
    for (int i = 0; i < num_boxes; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            boxes[i][j] = rawOutput[i * score_end_index + j];
        }
        for (int j = score_start_index; j < score_end_index; ++j)
        {
            scores[i][j - score_start_index] = rawOutput[i * score_end_index + j];
        }
    }
    std::cout << "[INFO] [Postprocess] Extract boxes and scores " << std::endl;

    // 边框缩放
    vector<vector<float>> xyxy_boxes = bbox_cxcywh_to_xyxy(boxes);

    // 判断是否需要归一化处理
    if (!is_normalized(scores)) {
        normalize_scores(scores);
    }

    // 找出每一组对应的最大值元素
    std::vector<float> max_scores;
    for (const auto& score_row : scores) {
        auto max_score = *std::max_element(score_row.begin(), score_row.end());
        max_scores.push_back(max_score);
    }

    // 过滤掉低于阈值的分数
    std::vector<bool> mask;
    for (const auto& max_score : max_scores) {
        mask.push_back(max_score > confThreshold);
    }

    // 根据结果筛选出符合条件的框
    std::vector<std::vector<float>> filtered_boxes, filtered_scores;
    for (std::size_t i = 0; i < xyxy_boxes.size(); ++i) {
        if (mask[i]) {
            filtered_boxes.push_back(xyxy_boxes[i]);
            filtered_scores.push_back(scores[i]);
        }
    }

    // 获取相应的类别ID
    std::vector<int> filtered_labels;
    std::vector<float> max_filtered_scores;
    for (const auto& score_row : filtered_scores) {
        auto max_score_it = std::max_element(score_row.begin(), score_row.end());
        auto max_score = *max_score_it;
        auto label = std::distance(score_row.begin(), max_score_it);
        filtered_labels.push_back(label);
        max_filtered_scores.push_back(max_score);
    }


    // 提取 boxes 中的 x1, y1, x2, y2
    std::vector<float> \
        x1(filtered_boxes.size()), y1(filtered_boxes.size()), \
        x2(filtered_boxes.size()), y2(filtered_boxes.size());
    for (int i = 0; i < filtered_boxes.size(); i++) {
        x1[i] = filtered_boxes[i][0];
        y1[i] = filtered_boxes[i][1];
        x2[i] = filtered_boxes[i][2];
        y2[i] = filtered_boxes[i][3];
    }

    // 对 x1, y1, x2, y2 进行缩放、取整、截断
    for (int i = 0; i < filtered_boxes.size(); i++) {
        x1[i] = std::floor(std::min(std::max(1.0f, x1[i] * imageWidth), imageWidth - 1.0f));
        y1[i] = std::floor(std::min(std::max(1.0f, y1[i] * imageHeight), imageHeight - 1.0f));
        x2[i] = std::ceil(std::min(std::max(1.0f, x2[i] * imageWidth), imageWidth - 1.0f));
        y2[i] = std::ceil(std::min(std::max(1.0f, y2[i] * imageHeight), imageHeight - 1.0f));
    }

    // 将 x1, y1, x2, y2 拼接回 boxes
    std::vector<std::vector<float>> new_boxes(filtered_boxes.size(), std::vector<float>(4));
    for (int i = 0; i < filtered_boxes.size(); i++) {
        new_boxes[i][0] = x1[i];
        new_boxes[i][1] = y1[i];
        new_boxes[i][2] = x2[i];
        new_boxes[i][3] = y2[i];
    }
    filtered_boxes = new_boxes;

    // 绘制结果图并保存
    draw_boxes_and_save_image(
        filtered_labels,
        max_filtered_scores,
        filtered_boxes,
        savePath,
        labels,
        imageBGR
    );
    std::cout << "[INFO] [Postprocess] Done! " << std::endl;

}