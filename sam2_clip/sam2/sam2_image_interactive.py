# use sam2-env

import cv2
from ultralytics import SAM
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

img_path = "../test_images/goods.png"
img = cv2.imread(img_path)
display_img = img.copy()

points = []

# 鼠标点击事件
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"点击点: ({x}, {y})")
        points.append([x, y])
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

def redraw_points():
    """重绘图片上的所有点"""
    global display_img
    display_img = img.copy()
    for (x, y) in points:
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

cv2.namedWindow("SAM Interactive")
cv2.setMouseCallback("SAM Interactive", click_event)

model = SAM("SAM_models/sam2.1_l.pt")
model.info()

print("左键点击添加点，按 'f' 分割，按 'r' 重置，按 'z' 撤销上一个点，按 'q' 退出。")

while True:
    cv2.imshow("SAM Interactive", display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('f'):  # Enter 键
        if points:
            labels = [1] * len(points)
            print(f"开始分割，使用点: {points}")
            results = model(img_path, points=points, labels=labels)

            result_img = results[0].plot()
            display_img = result_img.copy()
            points.clear()
            print("分割完成，可以继续点击新点。")
        else:
            print("未检测到点击点，请点击后再按 Enter。")

    elif key == ord('r'):  # 重置
        print("重置图片和点。")
        display_img = img.copy()
        points.clear()

    elif key == ord('z'):  # 撤销
        if points:
            removed_point = points.pop()
            print(f"撤销点: {removed_point}")
            redraw_points()
        else:
            print("没有点可以撤销。")

    elif key == ord('q'):  # 退出
        print("退出程序。")
        break

cv2.destroyAllWindows()
