"""
质量评估模块
评估每一步的质量, 给出整体置信度
"""
import numpy as np


def assess_contour_smoothness(contour):
    """
    评估轮廓光滑度: 计算相邻点间距的标准差
    越光滑的轮廓标准差越小
    返回: (is_smooth, score 0-1)
    """
    if len(contour) < 5:
        return False, 0.0

    diffs = np.diff(contour, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

    mean_dist = np.mean(distances)
    if mean_dist < 0.5:
        return False, 0.0

    std_dist = np.std(distances)
    cv = std_dist / (mean_dist + 1e-8)  # 变异系数

    if cv < 0.3:
        return True, 0.9
    elif cv < 0.6:
        return True, 0.7
    elif cv < 1.0:
        return False, 0.5
    else:
        return False, 0.3


def assess_symmetry(contour):
    """
    评估液滴左右对称性: 比较左右两侧点数
    返回: (is_symmetric, score 0-1)
    """
    if len(contour) < 10:
        return False, 0.0

    xs = contour[:, 0]
    apex_idx = np.argmin(contour[:, 1])  # 顶点 (y最小)
    x_apex = xs[apex_idx]

    left_count = np.sum(xs < x_apex)
    right_count = np.sum(xs > x_apex)

    if left_count == 0 or right_count == 0:
        return False, 0.0

    ratio = min(left_count, right_count) / max(left_count, right_count)

    if ratio > 0.8:
        return True, 0.9
    elif ratio > 0.5:
        return False, 0.6
    else:
        return False, 0.3


def assess_contour_coverage(contour, roi_width):
    """
    评估轮廓覆盖度: 轮廓跨度占 ROI 宽度的比例
    返回: score 0-1
    """
    if len(contour) < 5:
        return 0.0

    xs = contour[:, 0]
    x_range = np.max(xs) - np.min(xs)
    coverage = x_range / (roi_width + 1e-8)

    return min(coverage, 1.0)


def overall_quality(contour, roi_width, angle_left, angle_right):
    """
    综合质量评估
    返回: {
        'score': float (0-1),
        'level': 'high' | 'medium' | 'low',
        'details': dict
    }
    """
    _, smooth_score = assess_contour_smoothness(contour)
    _, sym_score = assess_symmetry(contour)
    cov_score = assess_contour_coverage(contour, roi_width)

    # 左右角度差异
    angle_diff = abs(angle_left - angle_right)
    if angle_diff < 3:
        angle_score = 1.0
    elif angle_diff < 8:
        angle_score = 0.7
    elif angle_diff < 20:
        angle_score = 0.4
    else:
        angle_score = 0.1

    # 加权综合
    score = 0.25 * smooth_score + 0.20 * sym_score + 0.15 * cov_score + 0.40 * angle_score

    if score > 0.75:
        level = "high"
    elif score > 0.45:
        level = "medium"
    else:
        level = "low"

    return {
        "score": round(score, 3),
        "level": level,
        "details": {
            "smoothness": round(smooth_score, 3),
            "symmetry": round(sym_score, 3),
            "coverage": round(cov_score, 3),
            "angle_consistency": round(angle_score, 3),
        },
    }
