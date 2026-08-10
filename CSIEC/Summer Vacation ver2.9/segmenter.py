"""
轮廓分割模块
逐列取最高点生成上轮廓, 保持列序 (x 递增)
角度计算时按自然列序 (不做 x 重排)
"""
import cv2
import numpy as np
from scipy.signal import savgol_filter


def _mask_to_profile(mask, baseline_guess):
    """二值掩膜 → 上轮廓 (每列最高白像素, x 递增)"""
    h, w = mask.shape
    profile = []
    for x in range(w):
        rows = np.where(mask[:, x] > 0)[0]
        if len(rows) > 0:
            y = int(rows[0])
            if baseline_guess is None or y < baseline_guess:
                profile.append([x, y])
    if len(profile) < 15:
        return None
    return np.array(profile, dtype=np.float64)


def _smooth_and_snake(profile, gray_img):
    """Savitzky-Golay 平滑 + Snake 精炼"""
    n = len(profile)
    wlen = min(31, max(7, n // 6))
    if wlen % 2 == 0:
        wlen += 1
    if n > wlen >= 5:
        profile[:, 1] = savgol_filter(profile[:, 1], wlen, 2)

    return snake_refine(profile.astype(int), gray_img,
                        max_iter=20, alpha=0.4, smooth_strength=0.35)


def snake_refine(contour_pts, gray_img, max_iter=25, alpha=0.5, smooth_strength=0.3):
    """活动轮廓精炼"""
    if len(contour_pts) < 5:
        return contour_pts
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(grad_x**2 + grad_y**2)
    edge = cv2.GaussianBlur(edge, (9, 9), 3)
    edge = edge / (edge.max() + 1e-8)
    h, w = gray_img.shape
    pts = contour_pts.astype(np.float64).copy()
    n_pts = len(pts)
    for _ in range(max_iter):
        new_pts = pts.copy()
        for i in range(n_pts):
            ip, in_ = max(0, i-1), min(n_pts-1, i+1)
            tx = pts[in_, 0] - pts[ip, 0]; ty = pts[in_, 1] - pts[ip, 1]
            tn = np.sqrt(tx**2 + ty**2)
            if tn < 0.5: continue
            nx, ny = -ty/tn, tx/tn
            best_s, best_p = -1., pts[i].copy()
            for step in range(-7, 8):
                sx = int(round(pts[i, 0] + step*nx))
                sy = int(round(pts[i, 1] + step*ny))
                if 0 <= sx < w and 0 <= sy < h and edge[sy, sx] > best_s:
                    best_s = edge[sy, sx]; best_p = np.array([sx, sy])
            new_pts[i] = pts[i] + alpha*(best_p - pts[i])
        for _ in range(3):
            sm = new_pts.copy()
            for i in range(1, n_pts-1):
                sm[i] = (1-smooth_strength)*new_pts[i] + smooth_strength/2*(new_pts[i-1]+new_pts[i+1])
            new_pts = sm
        if np.mean(np.sqrt(np.sum((new_pts-pts)**2, axis=1))) < 0.1: break
        pts = new_pts
    return np.round(pts).astype(int)


def _otsu(gray, guess, ci=2, oi=1):
    clean = gray.copy()
    if guess < clean.shape[0]: clean[guess:, :] = 0
    _, b = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((3,3), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=ci)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=oi)
    return b

def _adaptive(gray, guess):
    clean = gray.copy()
    if guess < clean.shape[0]: clean[guess:, :] = 0
    b = cv2.adaptiveThreshold(clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 5)
    k = np.ones((3,3), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=2)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)
    return b

def _canny(gray):
    e = cv2.Canny(gray, 20, 80)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=3)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(gray); cv2.drawContours(m, cnts, -1, 255, -1)
    return m


def segment_droplet(gray_roi, roi_baseline_guess):
    """多策略分割, 返回 (profile, mask, method)"""
    strategies = [
        ("otsu", lambda: _otsu(gray_roi, roi_baseline_guess)),
        ("adaptive", lambda: _adaptive(gray_roi, roi_baseline_guess)),
        ("canny", lambda: _canny(gray_roi)),
        ("otsu_loose", lambda: _otsu(gray_roi, roi_baseline_guess, ci=1, oi=0)),
    ]
    for name, fn in strategies:
        try:
            mask = fn()
            profile = _mask_to_profile(mask, roi_baseline_guess)
            if profile is not None and len(profile) >= 15:
                profile = _smooth_and_snake(profile, gray_roi)
                return profile, mask, name
        except Exception:
            continue
    raise RuntimeError("所有分割策略均失败")
