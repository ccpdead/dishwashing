#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()
def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
        # 显示图像
        cv2.imshow("RGB Image", cv_image)
        cv2.waitKey(1)

    except Exception as e:
        rospy.logerr(e)


def depth_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")#rgb8,mono8,passthrough
                # 将深度图像从浮点数类型转换为8位无符号整数类型
        print("中心点深度：{} mm".format(cv_image[200,320]))
        cv_image = cv2.normalize(cv_image, None, 0, 1000, cv2.NORM_MINMAX)
        
        
        cv_image = cv_image.astype(np.uint8)
        # 显示图像
        cv2.imshow("Depth Image", cv_image)
        cv2.imwrite("/home/jhr/depth.png",cv_image)
        cv2.waitKey(30)

    except Exception as e:
        rospy.logerr(e)
        


def main():
    rospy.init_node("image_viewer", anonymous=True)
    # rospy.Subscriber("/berxel_camera/rgb/rgb_raw", Image, image_callback)
    rospy.Subscriber("/berxel_camera/depth/depth_raw", Image, depth_callback)

    try:
        rospy.spin()# 等待信息
    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
