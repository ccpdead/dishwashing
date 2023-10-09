#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import tensorflow as tf

def image_callback(msg):
    try:
        # 使用cv_bridge将ROS图像消息转换为NumPy数组
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # 将NumPy数组转换为Tensor
        image_tensor = tf.convert_to_tensor(cv_image, dtype=tf.float32)

        # 现在您可以使用image_tensor进行TensorFlow的各种操作
        # 例如，您可以将其输入到神经网络中

        # 示例：打印图像张量的形状
        print("Image Tensor Shape:", image_tensor.shape)

    except Exception as e:
        rospy.logerr(e)

def main():
    rospy.init_node("image_processor", anonymous=True)
    rospy.Subscriber("/berxel_camera/rgb/rgb_raw", Image, image_callback)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
