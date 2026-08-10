"""
基线检测模块
核心思路: 接触点 = 轮廓切线角度的峰值位置

轮廓已按弧线顺序排列 (左触点→顶点→右触点),
沿弧线从顶点向两侧, 用弧长参数化计算局部切线角度,
角度峰值处 = 三相接触点。
"""
import cv2
import numpy as np
from scipy.signal import savgol_filter


def _local_angles_by_arc(arc_pts, window=8):
    """沿弧线逐点计算局部切线角度 (度)"""
    n = len(arc_pts)
    if n < 5:
        return np.zeros(n)

    angles = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        if hi - lo < 3:
            continue
        xw = arc_pts[lo:hi+1, 0]
        yw = arc_pts[lo:hi+1, 1]
        if np.max(xw) - np.min(xw) < 1.0:
            continue  # 近乎垂直, 跳过
        try:
            c = np.polyfit(xw, yw, 1)
            angles[i] = np.degrees(np.arctan(abs(c[0])))
        except np.linalg.LinAlgError:
            continue
    return angles


def _find_contact_side(seg):
    """
    在轮廓段找接触点: 切线角度峰值法

    seg: 轮廓段, 已按从顶点→表面的顺序排列
    返回: (contact_x, contact_y, contact_angle) 或 None
    """
    n = len(seg)
    if n < 10:
        return None

    raw = _local_angles_by_arc(seg)
    wlen = min(15, max(5, n // 6))
    if wlen % 2 == 0:
        wlen += 1
    ang = savgol_filter(raw, wlen, 2) if n > wlen >= 5 else raw.copy()

    pi = int(np.argmax(ang))
    pv = ang[pi]

    if pi >= n - int(n * 0.12):
        return (float(seg[-1, 0]), float(seg[-1, 1]), float(ang[-1]))
    if pv - np.min(ang[pi:]) > pv * 0.4:
        return (float(seg[pi, 0]), float(seg[pi, 1]), float(pv))
    return (float(seg[-1, 0]), float(seg[-1, 1]), float(ang[-1]))


def detect_contact_points(profile, roi_width, roi_height):
    """检测两个三相接触点"""
    if len(profile) < 10:
        return None, None, 0

    xs, ys = profile[:, 0], profile[:, 1]
    apex_idx = int(np.argmin(ys))

    # 左侧: profile 是 x 递增, 顶点之前的列需要反转 (从顶点→左边缘)
    left_seg = profile[:apex_idx][::-1] if apex_idx > 0 else np.array([])
    # 右侧: 顶点之后的列自然从顶点→右边缘
    right_seg = profile[apex_idx:]

    lc = _find_contact_side(left_seg) if len(left_seg) > 0 else None
    rc = _find_contact_side(right_seg) if len(right_seg) > 0 else None

    if lc and rc:
        by = (lc[1] + rc[1]) / 2.0
    elif lc:
        by = lc[1]
    elif rc:
        by = rc[1]
    else:
        by = np.max(ys)

    return lc, rc, by


def detect_baseline(contour, gray_roi, cx_guess, cy_guess, r_guess):
    """基线检测主函数"""
    h, w = gray_roi.shape
    lc, rc, by = detect_contact_points(contour, w, h)

    apex_y = np.min(contour[:, 1]) if len(contour) > 0 else 0
    if by < apex_y or by > h * 0.95:
        y_start = max(0, int(cy_guess + r_guess * 0.5))
        y_end = min(h, int(cy_guess + r_guess * 2.0))
        if y_start < y_end:
            roi_s = gray_roi[y_start:y_end, :]
            sy = cv2.Sobel(roi_s, cv2.CV_64F, 0, 1, ksize=3)
            by = int(np.argmax(np.sum(np.abs(sy), axis=1))) + y_start
            method = "gradient_fallback"
        else:
            by, method = cy_guess + r_guess, "guess_fallback"
        lc, rc = None, None
    else:
        method = "angle_peak"

    by = min(max(by, 1), h - 2)
    return int(by), method, (lc[0], lc[1]) if lc else None, (rc[0], rc[1]) if rc else None
