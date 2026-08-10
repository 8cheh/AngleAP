"""
图像预处理模块
处理顺序: 去噪 → 增强对比度 → 锐化
"""
import cv2
import numpy as np


def bilateral_denoise(img, d=9, sigma_color=75, sigma_space=75):
    """
    双边滤波: 保边去噪, 比高斯滤波更适合保留液滴边界
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def clahe_enhance(img, clip_limit=2.0, tile_size=8):
    """
    CLAHE 自适应直方图均衡化
    解决光照不均问题, 每个小格子独立均衡化再拼接
    """
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_clahe = clahe.apply(l_channel)
        merged = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(img)


def unsharp_mask(img, sigma=1.0, amount=0.5):
    """
    反锐化掩模: 原图 + amount * (原图 - 高斯模糊)
    增强边缘清晰度
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def preprocess(img, denoise=True, enhance=True, sharpen=True):
    """
    完整预处理管线
    返回: 处理后的图像
    """
    result = img.copy()

    if denoise:
        result = bilateral_denoise(result)

    if enhance:
        result = clahe_enhance(result)

    if sharpen:
        result = unsharp_mask(result, sigma=1.0, amount=0.5)

    return result
