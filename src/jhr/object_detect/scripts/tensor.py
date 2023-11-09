#!/usr/bin/env python3
import csv
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import threading
import time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
print("--------------------------------")
print("Python 环境路径:", sys.prefix)
print("Python 版本:", sys.version)
print("Python 可执行文件路径:", sys.executable)
print("--------------------------------")


name = {0: "cup", 1: "gcup", 2: "bowl", 3: "plate", 4: "spoon"}

# 加载模型，读取签名
new_model = tf.saved_model.load(
    "/home/jhr/Program/TensorFlow/00-model/model/last_saved_model-3")
model_signature = new_model.signatures["serving_default"]

depth_image = None
rgb_image = None


# 创建锁对象，用于同步访问全局图像变量
# 图像加载与模型检测
def compute(img):
    # start_time = time.time()
    if img is not None:
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
    else:
        print("img is None")

# 矩阵计算
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2.0  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2.0  # bottom right y
    return y

#0-cup,1-gcup,2-bowl,3-plate,4-spoon
#cup 55-65
#bowl 70-80
#plate 105-120
def class_detection(Rect):
    output_Rect = np.empty([len(Rect),6],dtype=float)
    for i in range(len(Rect)):
        x = int(Rect[i, 0])
        y = int(Rect[i, 1])
        x2 = int(Rect[i, 2])
        y2 = int(Rect[i, 3])
        class_name = int(Rect[i,-1])#类别
        judge_class=0
        #如果检测到是cup，bowl，以及plate，根据尺寸进行第二次判断
        if(class_name==0 or class_name==2 or class_name==3):
            max_length = max(int(x2-x),int(y2-y))#计算最大边长
            if class_name == 0:#cup
                if max_length>55 and max_length<65:
                    judge_class=0
                elif max_length>66 and max_length<88:
                    judge_class=2
                elif max_length>105 and max_length<120:
                    judge_class=3
                    
            elif class_name == 2:#bowl
                if max_length>55 and max_length<65:
                    judge_class=0
                elif max_length>70 and max_length<80:
                    judge_class=2
                elif max_length>105 and max_length<120:
                    judge_class=3
                    
            elif class_name == 3:#plate
                if max_length>55 and max_length<65:
                    judge_class=0
                elif max_length>70 and max_length<80:
                    judge_class=2
                elif max_length>105 and max_length<120:
                    judge_class=3
        else:
            judge_class = class_name
        #返回位置，种类，以及置信度
        for x in range(0,4):
            output_Rect[i,x]=int(Rect[i, x])
        output_Rect[i,-1] = judge_class #修改Rect
        output_Rect[i,-2]=Rect[i,-2]
    return output_Rect


# 正反检测
def Positive_negative_detection(Rect, input_src, input_rgb):
    if input_src is not None:
        for i in range(len(Rect)):
            if int(Rect[i, -1]) == 2 or int(Rect[i, -1]) == 3:
                x = int(Rect[i, 0])
                y = int(Rect[i, 1])
                x2 = int(Rect[i, 2])
                y2 = int(Rect[i, 3])

                # 创建掩膜
                src = input_src[y:y2, x:x2]
                mask = np.zeros_like(src, dtype=np.uint8)
                mask2 = np.zeros_like(src, dtype=np.uint8)

                # 计算最小圆心
                h, w = src.shape
                com1 = int((x2 - x) / 2)
                com2 = int((y2 - y) / 2)
                if com1 >= com2:
                    radius = com2
                else:
                    radius = com1

                # 绘制外部圆
                cv2.circle(mask2,
                           (int(w/2), int(h/2)),
                           int(radius/6)*6,
                           (255, 255, 255),
                           -1)
                # 绘制内部圆
                cv2.circle(mask,
                           (int(w/2), int(h/2)),
                           int(radius/6)*3,
                           (255, 255, 255),
                           -1)
                                
                mask_image = cv2.bitwise_and(src, mask)
                mask2 = mask2-mask  # 对像素进行算术运算
                mask_image2 = cv2.bitwise_and(src, mask2)

                # 计算内环像素平均值
                count = 0
                dep_sum = 0
                input = 0
                output = 0
                max = 0
                min = 0
                if mask_image.any() and mask_image2.any():
                    max = mask_image.max()
                    min = mask_image.min()

                    mask = (mask_image > min) & (mask_image < max)
                    dep_sum = np.sum(mask_image[mask])
                    count = np.sum(mask)
                    if count > 0:
                        input = int(dep_sum / count)
                    else:
                        print("no data")

                    # 计算外环平均值
                    max = mask_image2.max()
                    min = mask_image2.min()
                    count = 0
                    dep_sum = 0

                    mask = (mask_image2 > min) & (mask_image2 < max)
                    dep_sum = np.sum(mask_image2[mask])
                    count = np.sum(mask)

                    if count > 0:
                        output = int(dep_sum / count)
                    else:
                        print("no data")
                    if input != 0 or output != 0:
                        if input < output:
                            cv2.putText(input_rgb, "back", (int(Rect[i, 0])+55, int(
                                Rect[i, 1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_4)
                        else:
                            cv2.putText(input_rgb, "forward", (int(Rect[i, 0])+55, int(
                                Rect[i, 1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_4)
                    
        return input_rgb
    else:
        print("depth is Zero")

# 数据显示
def img_show(Rect, src):
    if src is not None:
        for i in range(len(Rect)):

            x = int(Rect[i, 0])
            y = int(Rect[i, 1])
            x2 = int(Rect[i, 2])
            y2 = int(Rect[i, 3])
            cv2.putText(src,"({})".format(max(int(x2-x),int(y2-y))),(int(Rect[i,0]),int(Rect[i,1]+20)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,200),2,cv2.LINE_4)
            # 绘制圆
            radius = min(int((x2-x)/2), int((y2-y)/2))
            for r in range(1, 7, 1):
                cv2.circle(src,
                           (int((x+x2)/2), int((y+y2)/2)),
                           (int(radius/6)*r),
                           (0, 255, 0),
                           1)
            # 绘制矩形框
            cv2.rectangle(src, (x, y), (x2, y2), (0, 255, 0), 2)
            # 显示类别
            cv2.putText(src,
                        name[int(Rect[i, -1])],
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_4)
            # 显示概率
            cv2.putText(src,
                        "{:.2}%".format(Rect[i, -2]),
                        (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                        cv2.LINE_4)
        cv2.imshow("object_detect", src)
        cv2.waitKey(60)
    else:
        print("rgb is Zero")

# 非极大值抑制
def non_max_suppression(resout, frame, depth, conf_thres=0.5, iou_thres=0.5, mi=10):
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
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = tf.image.non_max_suppression(
            boxes, scores, iou_threshold=iou_thres, max_output_size=15)
        i = i[:max_det]  # limit detections
        output[xi] = tf.gather(x, i)

    #正反检测
    img = Positive_negative_detection(class_detection(output[0]), depth, frame)
    # 显示图像
    img_show(class_detection(output[0]), img)  

#image回调函数
def image_callback(msg):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(cv_image, (640, 400),
                           interpolation=cv2.INTER_LINEAR)
        global rgb_image
        rgb_image = frame
        # cv2.imshow("input_rgb", rgb_image)
    except Exception as e:
        rospy.logerr(e)

#depth回调函数
def depth_callback(msg):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
        # 将深度图像从浮点数类型转换为8位无符号整数类型
        cv_image = cv2.normalize(cv_image, None, 0, 1024, cv2.NORM_MINMAX)
        cv_image = cv_image.astype(np.uint8)
        global depth_image
        depth_image = cv_image
        # cv2.imshow("input_depth", cv_image)
    except Exception as e:
        rospy.logerr(e)


def thread_job():
    rospy.spin()


def main():
    rospy.init_node("tensor_detect", anonymous=True)

    add_thread = threading.Thread(target=thread_job)
    add_thread.start()

    rospy.Subscriber("/berxel_camera/rgb/rgb_raw", Image,
                     image_callback)
    rospy.Subscriber("/berxel_camera/depth/depth_raw",
                     Image, depth_callback)
    time.sleep(1)

    while (1):
        time.sleep(0.03)
        resout = compute(rgb_image)  # 模型预测
        non_max_suppression(resout, rgb_image, depth_image)  # 非极大值抑制

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
