'''
Author: CVHub
Date: May 2023
Description: The project demonstrates how to deploy the RT-DETR model in Python using ONNXRUNTIME.
'''

import cv2
import random
import numpy as np
import onnxruntime as ort


random.seed(10086)
CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
CLASS_COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASS_NAMES))]


def bbox_cxcywh_to_xyxy(boxes):

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return np.stack([x1, y1, x2, y2], axis=1)


def init_resources(model_path):

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    providers = ort.get_available_providers()
    ort_model = ort.InferenceSession(
        model_path,
        prividers=providers,
        sess_options=sess_options
    )

    return ort_model


def preprocess(image_path, ort_model):

    # 加载图片
    image = cv2.imread(image_path)

    # 获取图像宽高
    image_h, image_w = image.shape[:2]

    # 获取所有输入节点信息
    input_nodes = ort_model.get_inputs()

    # 筛选出名称为"images"的输入节点信息
    input_node = next(filter(lambda n: n.name == "image", input_nodes), None)

    if input_node is None:
        raise ValueError("No input node with name 'image'")

    # 输入尺寸
    input_shape = input_node.shape[-2:]
    input_h, input_w = input_shape

    # 缩放因子
    ratio_h = input_h / image_h
    ratio_w = input_w / image_w
    
    # 预处理步骤
    img = cv2.resize(image, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=2)
    img = img[:, :, ::-1] / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img, dtype=np.float32)

    return image, img, (image_h, image_w)


def postprocess(outs, conf_thres, im_shape):

    boxes, scores = outs[:, :4], outs[:, 4:]

    # 根据 scores 数值分布判断是否进行归一化处理
    if not (np.all((scores > 0) & (scores < 1))):
        print("normalized value >>>")
        scores = 1 / (1 + np.exp(-scores))

    boxes = bbox_cxcywh_to_xyxy(boxes)
    _max = scores.max(-1)
    _mask = _max > conf_thres
    boxes, scores = boxes[_mask], scores[_mask]
    labels, scores = scores.argmax(-1), scores.max(-1)

    # 对边框信息进行尺度归一化
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = np.floor(np.minimum(np.maximum(1, x1 * im_shape[1]), im_shape[1] - 1)).astype(int)
    y1 = np.floor(np.minimum(np.maximum(1, y1 * im_shape[0]), im_shape[0] - 1)).astype(int)
    x2 = np.ceil(np.minimum(np.maximum(1, x2 * im_shape[1]), im_shape[1] - 1)).astype(int)
    y2 = np.ceil(np.minimum(np.maximum(1, y2 * im_shape[0]), im_shape[0] - 1)).astype(int)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    return labels, scores, boxes


def main():

    # 参数初始化
    conf_thres = 0.45
    image_path = '../images/bus.jpg'
    save_path = '../images/bus_python_result.jpg'
    model_path = '../weights/rtdetr_r50vd_6x_coco_cvhub.onnx'

    # 资源初始化
    ort_model = init_resources(model_path)

    # 预处理
    im0, blob, im_shape = preprocess(image_path, ort_model)

    # 模型推理
    outs = ort_model.run(None, {'image': blob})[0][0]

    # 后处理
    labels, scores, boxes = postprocess(outs, conf_thres, im_shape)

    # 保存结果
    for label, score, box in zip(labels, scores, boxes):
        label_text = f'{CLASS_NAMES[label]}: {score:.2f}'
        cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), CLASS_COLORS[label], 2)
        cv2.putText(im0, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(save_path, im0)


if __name__ == '__main__':
    main()