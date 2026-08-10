"""
液滴定位模块
找到液滴的大致中心位置和半径, 为后续 ROI 裁剪和分割初始化提供依据
"""
import cv2
import numpy as np


def locate_by_hough(gray_img, param1=50, param2=25):
    """
    霍夫圆检测定位
    对噪声容忍度高, 适合圆形/近圆形液滴
    返回: (cx, cy, r) 或 None
    """
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1.0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=param1,
        param2=param2,
        minRadius=5,
        maxRadius=300,
    )

    if circles is not None:
        # 取第一个检测到的圆 (circles shape: (1, N, 3))
        cx, cy, r = np.round(circles[0, 0, :])
        return (int(cx), int(cy), int(r))
    return None


def locate_by_edge_density(gray_img):
    """
    边缘密度定位 (霍夫圆失败时的备选)
    对全图做 Canny, 距离变换找到边缘最密集区域即为液滴中心
    """
    edges = cv2.Canny(gray_img, 30, 100)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    dist_blur = cv2.GaussianBlur(dist, (31, 31), 0)

    _, _, _, max_loc = cv2.minMaxLoc(dist_blur)
    cx, cy = max_loc

    h, w = gray_img.shape
    radius = min(cx, cy, w - cx, h - cy) // 2

    return (cx, cy, radius)


def locate_droplet(gray_img):
    """
    液滴定位主函数, 含自动降级策略
    返回: ((cx, cy, r), method_name)
    """
    # 策略 1: 标准霍夫圆
    result = locate_by_hough(gray_img)
    if result is not None:
        return result, "hough"

    # 策略 2: 放宽参数的霍夫圆
    result = locate_by_hough(gray_img, param1=30, param2=15)
    if result is not None:
        return result, "hough_relaxed"

    # 策略 3: 边缘密度定位
    result = locate_by_edge_density(gray_img)
    return result, "edge_density"
