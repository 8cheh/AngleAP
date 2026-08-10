"""
角度计算模块
双方法并行 + 返回曲线数据供可视化

  Y-L 全局拟合: 有物理依据, 返回 ODE 解曲线
  局部接触点拟合: 在接触点附近取窗口做多项式, 返回拟合曲线
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp


# ================================================================
#  坐标转换
# ================================================================

def _prepare_contour(contour, baseline_y):
    """轮廓 → (x, z), z 向上为正, 顶点为原点"""
    xs = contour[:, 0].astype(np.float64)
    ys = contour[:, 1].astype(np.float64)
    zs = (baseline_y - ys).astype(np.float64)

    apex_idx = np.argmax(zs)
    z_apex = zs[apex_idx]

    keep = zs > z_apex * 0.15
    xs, zs = xs[keep], zs[keep]
    if len(xs) < 5:
        raise ValueError(f"有效轮廓点太少 ({len(xs)})")

    apex_idx = np.argmax(zs)
    return xs - xs[apex_idx], zs - zs[apex_idx]


def _split_sides(xc, zc):
    left = xc < 0; right = xc > 0
    return (-xc[left], zc[left]), (xc[right], zc[right])


# ================================================================
#  Y-L 方程及拟合
# ================================================================

def _yl_ode(s, y, b):
    x, z, phi = y
    if x < 1e-8:
        return [1., 0., 1. / b]
    return [np.cos(phi), np.sin(phi), 2. / b - np.sin(phi) / x]


def _fit_yl_one_side(x_side, z_side):
    """
    Y-L 单侧拟合
    返回: (angle_deg, rmse, curve_data) 或 None
    curve_data: (x_curve, z_curve) -- 拟合曲线在图像坐标中
    """
    if len(x_side) < 8:
        return None
    si = np.argsort(x_side)
    xs = x_side[si]; zs = z_side[si]
    zp = -zs

    def obj(p):
        bv = p[0]
        if bv <= 0: return np.full(len(xs), 1e6)
        try:
            sol = solve_ivp(_yl_ode, (0, np.max(xs) * 1.5), [1e-8, 0., 1e-8],
                            args=(bv,), method='RK45', dense_output=True, rtol=1e-4, atol=1e-6)
            if not sol.success or len(sol.t) < 5: return np.full(len(xs), 1e6)
            return sol.sol(xs)[1] - zp
        except Exception:
            return np.full(len(xs), 1e6)

    rg = (np.max(xs)**2 + np.max(zp)**2) / (2 * np.max(zp) + 1e-8)
    try:
        res = least_squares(obj, [max(rg, 5)], bounds=([1.], [np.inf]), max_nfev=200)
        rmse = np.sqrt(np.mean(obj([res.x[0]])**2))
        sol = solve_ivp(_yl_ode, (0, np.max(xs)), [1e-8, 0., 1e-8],
                        args=(res.x[0],), method='RK45', rtol=1e-4, atol=1e-6)
        ang = max(0, min(180, np.degrees(sol.y[2, -1])))

        # 生成曲线点用于可视化
        x_curve = np.linspace(0, np.max(xs), 50)
        z_curve = sol.sol(x_curve)[1]
        curve = (x_curve, z_curve)

        return ang, rmse, curve
    except Exception:
        return None


# ================================================================
#  局部多项式拟合
# ================================================================

def _fit_poly_local(x_side, z_side, window=25, degree=3):
    """
    接触点附近窗口多项式拟合
    返回: (angle_deg, rmse, (x_fit, z_fit, coeffs)) 或 None
    """
    if len(x_side) < 5:
        return None
    si = np.argsort(x_side)
    xs = x_side[si]; zs = z_side[si]

    n = min(window, len(xs))
    xl = xs[-n:]; zl = zs[-n:]

    if len(xl) < 4:
        return None
    try:
        deg = min(degree, len(xl) - 1)
        coeffs = np.polyfit(xl, zl, deg)
        slope = np.polyval(np.polyder(coeffs), xl[-1])
        ang = max(0, min(180, np.degrees(np.arctan(abs(slope)))))
        rmse = np.sqrt(np.mean((zl - np.polyval(coeffs, xl))**2))
        x_fit = np.linspace(xl[0], xl[-1], 30)
        z_fit = np.polyval(coeffs, x_fit)
        return ang, rmse, (x_fit, z_fit, coeffs)
    except Exception:
        return None


# ================================================================
#  主计算函数
# ================================================================

def calculate_contact_angle(contour, baseline_y):
    """
    使用 contour 和 baseline 计算接触角

    返回:
    {
        'angle_yl': float | None,        # Y-L 结果
        'angle_local': float | None,     # 局部拟合结果
        'angle_avg': float,              # 主结果
        'angle_left': float, 'angle_right': float,
        'method': str,
        'confidence': float,
        'curves': {
            'yl_left': (x, z) | None,    # Y-L 左曲线
            'yl_right': (x, z) | None,   # Y-L 右曲线
            'poly_left': (x_fit, z_fit) | None,   # 多项式左曲线
            'poly_right': (x_fit, z_fit) | None,  # 多项式右曲线
        },
    }
    """
    xc, zc = _prepare_contour(contour, baseline_y)
    (xl, zl), (xr, zr) = _split_sides(xc, zc)

    curves = {"yl_left": None, "yl_right": None, "poly_left": None, "poly_right": None}

    # ---- Y-L ----
    yl_l = _fit_yl_one_side(xl, zl)
    yl_r = _fit_yl_one_side(xr, zr)
    yl_ok = (yl_l and yl_r and yl_l[1] < 10 and yl_r[1] < 10
             and 0 < yl_l[0] < 180 and 0 < yl_r[0] < 180)
    if yl_ok:
        curves["yl_left"] = yl_l[2]
        curves["yl_right"] = yl_r[2]

    # ---- 局部多项式 ----
    poly_l = _fit_poly_local(xl, zl)
    poly_r = _fit_poly_local(xr, zr)
    if poly_l and poly_r:
        curves["poly_left"] = poly_l[2][:2]   # (x_fit, z_fit)
        curves["poly_right"] = poly_r[2][:2]

    # ---- 汇总 ----
    if yl_ok:
        al, ar = yl_l[0], yl_r[0]
    elif poly_l and poly_r:
        al, ar = poly_l[0], poly_r[0]
    else:
        al, ar = 90., 90.

    avg = (al + ar) / 2
    method = "YL" if yl_ok else "LocalFit"

    diff = abs(al - ar)
    conf = 0.95 if diff < 3 else 0.80 if diff < 8 else 0.60 if diff < 15 else 0.40

    return {
        "angle_yl": ((yl_l[0] + yl_r[0]) / 2) if yl_ok else None,
        "angle_local": ((poly_l[0] + poly_r[0]) / 2) if (poly_l and poly_r) else None,
        "angle_avg": avg,
        "angle_left": al,
        "angle_right": ar,
        "method": method,
        "confidence": conf,
        "curves": curves,
        "yl_ok": yl_ok,
    }
