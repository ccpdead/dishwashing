#!/usr/bin/env python3
import csv
import cv2
import numpy as np
import tensorflow as tf

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
print("--------------------------------")
print("Python 环境路径:", sys.prefix)
print("Python 版本:", sys.version)
print("Python 可执行文件路径:", sys.executable)
print("--------------------------------")
csv_file = "/home/jhr/depth.csv"

name = {0: "cup", 1: "gcup", 2: "bowl", 3: "plate", 4: "spoon"}
# 加载模型，读取签名
new_model = tf.saved_model.load(
    "/home/jhr/Program/TensorFlow/00-model/model/last_saved_model-3")
model_signature = new_model.signatures["serving_default"]

depth_image = None

# 图像加载与模型检测


def compute(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [640, 640])
    img = img / 255.0
    dataset = tf.constant(img)  # IMG转化为Tensor
    dataset = np.expand_dims(dataset, 0)  # 扩充维度
    resout = model_signature(x=dataset)  # 模型预测
    resout = resout["output_0"]
    resout = np.array(resout)
    resout[0][..., :4] *= [640, 400, 640, 400]
    return resout

# 矩阵计算


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2.0  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2.0  # bottom right y
    return y

# 矩形框显示


def img_show(Rect, src):
    if src is not None:
        with open(csv_file, mode='w', newline="") as file:
            writer = csv.writer(file)
            for i in range(len(Rect)):

                x = int(Rect[i, 0])
                y = int(Rect[i, 1])
                x2 = int(Rect[i, 2])
                y2 = int(Rect[i, 3])
                c = name[int(Rect[i, -1])]
                p = int(Rect[i, -2])
                data = [x, y, x2, y2, c, p]
                writer.writerow(data)

                roi = src[y:y2, x:x2]
                # 绘制圆心
                cv2.circle(src,
                           (int((x+x2)/2), int((y+y2)/2)),
                           3,
                           (0, 255, 0),
                           -1)
                # 绘制矩形框
                cv2.rectangle(src, (x, y), (x2, y2), (0, 255, 0), 2)
                # 显示类别
                cv2.putText(src,
                            name[int(Rect[i, -1])],
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)
                # 显示概率
                cv2.putText(src,
                            "{:.2}%".format(Rect[i, -2]),
                            (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                            cv2.LINE_4)

                # cv2.imshow("{}".format(i), roi)
            cv2.imshow("object_detect", src)
            cv2.imwrite("/home/jhr/depth_image2.png",src)

            cv2.waitKey(1)
    else:
        print("Failed to load image .")

# 非极大值抑制


def non_max_suppression(resout, frame, conf_thres=0.5, iou_thres=0.5, mi=10):

    max_wh = 7680
    max_nms = 30000
    max_det = 300
    bs = resout.shape[0]  # batch_size
    nc = resout.shape[2] - 5  # number of classes
    xc = resout[..., 4] > conf_thres  # condidates
    output = [tf.zeros((0, 6))] * bs
    for xi, x in enumerate(resout):
        x = x[xc[xi]]
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])  # 计算四个边框
        mask = x[:, mi:]

        conf = tf.reduce_max(x[:, 5:mi], axis=1, keepdims=True)  # 计算每一列最大值
        j = np.expand_dims(tf.math.argmax(
            x[:, 5:mi], axis=1), -1)  # 保存每列最大值的索引

        concatenated = tf.concat([box, conf, j, mask], axis=1)
        conf_mask = tf.reshape(conf, [-1]) > conf_thres  # 展平矩阵，并与conf_thres比较
        # center_x, center_y, width,
        x = tf.boolean_mask(concatenated, conf_mask, axis=0)

        sorted_indices = tf.argsort(x[:, 4], direction="DESCENDING")
        x = tf.gather(x, sorted_indices[:max_nms])

        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = tf.image.non_max_suppression(
            boxes, scores, iou_threshold=iou_thres, max_output_size=15)
        i = i[:max_det]  # limit detections
        output[xi] = tf.gather(x, i)
    # img_show(output[0], frame)#显示图像
    img_show(output[0], depth_image)


def image_callback(msg):
    try:
        # 使用cv_bridge将ROS图像消息转换为OpenCV格式
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")

        frame = cv2.resize(cv_image, (640, 400),
                           interpolation=cv2.INTER_LINEAR)

        resout = compute(frame)  # 模型预测
        non_max_suppression(resout, frame)  # 非极大值抑制

    except Exception as e:
        rospy.logerr(e)


def depth_callback(msg):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
        # 将深度图像从浮点数类型转换为8位无符号整数类型
        cv_image = cv2.normalize(cv_image, None, 0, 1000, cv2.NORM_MINMAX)
        cv_image = cv_image.astype(np.uint8)

        # # 增强显示深度图像
        # enhanced_depth_image = cv2.applyColorMap(cv_image, cv2.COLORMAP_JET)

        # cv2.imshow("depth",cv_image)
        cv2.imwrite("/home/jhr/depth_image1.png", cv_image)
        global depth_image
        depth_image = cv_image
        cv2.waitKey(30)
    except Exception as e:
        rospy.logerr(e)


def main():
    rospy.init_node("tensor_detect", anonymous=True)
    rospy.Subscriber("/berxel_camera/depth/depth_raw", Image, depth_callback)
    rospy.Subscriber("/berxel_camera/rgb/rgb_raw", Image, image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.....\n")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()