"""
完整管线编排
支持自动模式和用户选点模式
"""
import cv2
import numpy as np
import time

from preprocessor import preprocess
from localizer import locate_droplet
from segmenter import segment_droplet
from baseline import detect_baseline
from calculator import calculate_contact_angle
from quality import overall_quality


class ContactAnglePipeline:

    def __init__(self):
        self._stage1_cache = {}

    def detect(self, image_path):
        """
        阶段1: 预处理 + 定位 + 分割, 返回 ROI 和轮廓供用户选点

        返回:
        {
            'roi': np.ndarray (BGR),      # ROI 增强图像
            'contour': np.ndarray,         # 轮廓点 (x, y), ROI 坐标
            'cx': int, 'cy': int,         # 液滴中心 (原始图像坐标)
            'radius': int,                 # 液滴半径
            'roi_offset': (x1, y1),       # ROI 在原图中的偏移
            'error': str | None,
        }
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"无法读取: {image_path}"}

        try:
            enhanced = preprocess(img)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            (cx, cy, r), _ = locate_droplet(gray)

            margin = r * 2
            x1, y1 = max(0, cx - margin), max(0, cy - margin)
            x2 = min(enhanced.shape[1], cx + margin)
            y2 = min(enhanced.shape[0], cy + margin)

            roi = enhanced[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cx_r, cy_r = cx - x1, cy - y1

            guess = min(int(cy_r + r * 1.2), roi_gray.shape[0] - 2)
            contour, _, seg_method = segment_droplet(roi_gray, guess)

            result = {
                "roi": roi,
                "contour": contour,
                "cx": cx, "cy": cy, "radius": r,
                "roi_offset": (x1, y1),
                "seg_method": seg_method,
            }

            # 缓存供 stage2 使用
            self._stage1_cache[image_path] = result
            return result
        except Exception as e:
            return {"error": str(e)}

    def calculate(self, image_path, left_pt, right_pt):
        """
        阶段2: 用户选定接触点后, 计算接触角

        left_pt, right_pt: (x, y) ROI 坐标

        返回:
        {
            'angle': float, 'angle_yl': float, 'angle_local': float,
            'angle_left': float, 'angle_right': float,
            'method': str, 'confidence': float,
            'curves': {...},             # 拟合曲线数据
            'baseline_y': float,
            'error': str | None,
        }
        """
        t0 = time.time()

        cached = self._stage1_cache.get(image_path)
        if cached is None:
            return {"error": "请先执行阶段1 (detect)"}

        try:
            roi = cached["roi"]
            contour = cached["contour"]
            h, w = roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)[:2]

            # 基线 = 两个用户选定接触点连线的 y 平均值
            baseline_y = (left_pt[1] + right_pt[1]) / 2.0

            # 只保留基线及以上的轮廓 (含接触点)
            above = contour[:, 1] <= baseline_y + 1
            valid = contour[above]
            if len(valid) < 5:
                return {"error": "基线以上轮廓点太少"}

            # 计算角度
            ang = calculate_contact_angle(valid, baseline_y)

            return {
                "angle": ang["angle_avg"],
                "angle_yl": ang["angle_yl"],
                "angle_local": ang["angle_local"],
                "angle_left": ang["angle_left"],
                "angle_right": ang["angle_right"],
                "method": ang["method"],
                "confidence": ang["confidence"],
                "curves": ang["curves"],
                "yl_ok": ang["yl_ok"],
                "baseline_y": baseline_y,
                "left_contact": left_pt,
                "right_contact": right_pt,
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            return {"error": str(e)}

    def process(self, image_path):
        """全自动模式 (向后兼容)"""
        r = self.detect(image_path)
        if r.get("error"):
            return {"success": False, "error": r["error"],
                    "metadata": {"elapsed_ms": 0}}

        roi = r["roi"]
        contour = r["contour"]
        cx_r = r["cx"] - r["roi_offset"][0]
        cy_r = r["cy"] - r["roi_offset"][1]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 自动基线
        baseline_y, _, left_pt, right_pt = detect_baseline(
            contour, roi_gray, cx_r, cy_r, r["radius"]
        )

        above = contour[:, 1] <= baseline_y + 1
        valid = contour[above]

        if len(valid) < 5:
            return {"success": False, "error": "基线以上轮廓点太少",
                    "metadata": {"elapsed_ms": 0}}

        ang = calculate_contact_angle(valid, baseline_y)

        quality = overall_quality(valid, roi_gray.shape[1],
                                  ang["angle_left"], ang["angle_right"])

        return {
            "success": True,
            "roi_image": roi,
            "contour": valid,
            "angle": ang["angle_avg"],
            "angle_yl": ang["angle_yl"],
            "angle_local": ang["angle_local"],
            "angle_left": ang["angle_left"],
            "angle_right": ang["angle_right"],
            "method": ang["method"],
            "confidence": ang["confidence"],
            "confidence_level": quality["level"],
            "confidence_score": quality["score"],
            "curves": ang["curves"],
            "yl_ok": ang["yl_ok"],
            "baseline_y_roi": baseline_y,
            "left_contact": left_pt,
            "right_contact": right_pt,
            "steps": {
                "segmentation": r["seg_method"],
                "baseline": "auto",
                "calculation": ang["method"],
            },
            "metadata": {
                "cx": r["cx"], "cy": r["cy"], "radius": r["radius"],
                "contour_points": len(valid),
                "elapsed_ms": 0,
            },
        }
