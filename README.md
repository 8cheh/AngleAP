# AngleAP — Automated Contact Angle Measurement Platform

## 自动化接触角测量平台

---

### 📖 Introduction | 项目简介

AngleAP is an automated contact angle measurement platform developed for CUPET. It takes droplet images as input and outputs accurate contact angle values through a fully automated pipeline, including droplet localization, image enhancement, segmentation, and Young-Laplace equation fitting.

AngleAP 是为 CUPET 开发的自动化接触角测量平台。它以液滴图像为输入，通过全自动流水线（液滴定位 → 图像增强 → 分割 → Young-Laplace 方程拟合）输出精确接触角数值。

> **Program developed by 8cheh, in collaboration with Qwen (my favorite AI).**
>
> 程序由 8cheh 开发，Qwen（我最喜欢的 AI）协作完成。

---

### 📁 Project Structure | 项目结构

```
AngleAP/
├── AP_app.py                # Flask Web Application / Flask Web 应用主程序
├── AP_Picbetter.py          # Image Preprocessing & Enhancement / 图像预处理增强模块
├── AP_Segmenter.py          # Droplet Segmentation / 液滴分割模块
├── AP_YLCalculator.py       # Young-Laplace Solver & Contact Angle Calculation / Young-Laplace 求解与接触角计算
├── Test/
│   ├── TE_contact_angle.py  # Manual Test Tool (click 2 points) / 手动选点测试工具
│   └── drop_image.jpg       # Sample Test Image / 测试图片
└── Enhancer/
    └── freeenhancer.py      # Advanced Image Enhancement (developing) / 高级增强模块（开发中）
```

---

### 🔬 Core Methods | 核心方法

Step ：
| 1 | **Droplet Localization** — Automatically locate the droplet position and size using the Hough Circle Transform algorithm. 
| **液滴定位** — 基于霍夫圆检测算法，自动识别图像中液滴的位置与尺寸。 |
| 2 | **Image Enhancement** — Improve droplet outline contrast and clarity using CLAHE combined with Gaussian sharpening. 
| **图像增强** — 结合 CLAHE（限制对比度自适应直方图均衡化）与高斯锐化，提升液滴轮廓的对比度与清晰度。 |
| 3 | **Droplet Segmentation** — Separate droplet from background using Otsu's adaptive thresholding and morphological operations; intelligently detect the solid-liquid interface baseline. 
| **液滴分割** — 利用 Otsu 自适应阈值分割与形态学操作分离液滴与背景，并智能识别固-液界面基线。 |
| 4 | **Contact Angle Calculation** — Numerically fit the droplet profile using the Young-Laplace equation (Runge-Kutta ODE solver + least-squares optimization); compute left/right contact angles and average them. 
| **接触角计算** — 基于 Young-Laplace 方程对液滴轮廓进行数值拟合（龙格-库塔法求解 ODE + 最小二乘优化）；分别计算左右接触角并取平均值。 |

复杂度
  ↑
  
  │  v1.7.py          ← 最完整：AI(SAM)+传统CV+Young-Laplace
  
  │  v4.4.py          ← 完整自动化：多尺度Hough+传统CV+Young-Laplace
  
  │  v3.7app.py       ← 3.7 的 Web 封装版（Flask 在线测量）
  
  │  v3.7picbetter.py ← 半自动流水线主程序
  
  │  v3.7segmenter.py ← 只做「分割 + 基准线」
  
  │  v3.7YLCal.py     ← 只做「Young-Laplace 拟合」
  
  │  v2.2.py          ← 手动点选 + 多项式拟合
  
  │  v1.py (Matlab)   ← 最原始：三点画圆求接触角
  
  └──────────────────────────────────────────────→ 自动化程度


一、整体概览（Overview）

这套程序围绕接触角自动测量（Contact Angle Measurement）展开，涵盖了从最基础的教学方法到AI + 物理模型的高级方案，体现了研究思路的逐步演进。

This collection of programs focuses on contact angle measurement, covering approaches from basic educational methods to advanced AI + physics-based solutions, reflecting a progressive research methodology.

二、程序分类（Classification）

按自动化程度与复杂度可分为四类：
1. 手动方法（Manual）：1.py（Matlab）、2.2.py  
2. 模块化工具（Modular Tools）：3.7segmenter.py、3.7YLCal.py  
3. 半自动流水线（Semi-automatic Pipeline）：3.7picbetter.py、3.7app.py  
4. 全自动系统（Fully Automatic Systems）：4.4.py、1.7.py

Based on automation level and complexity:
1. Manual methods: 1.py (Matlab), 2.2.py  
2. Modular tools: 3.7segmenter.py, 3.7YLCal.py  
3. Semi-automatic pipelines: 3.7picbetter.py, 3.7app.py  
4. Fully automatic systems: 4.4.py, 1.7.py

三、逐程序详解（Detailed Explanation）

1️⃣ 1.py（Matlab）—— 经典三点圆法

• 手动在图像上选取 3 个点

• 利用三点确定一个圆

• 接触角由几何关系计算：  

  \( \theta = 90^\circ - \arcsin\left(\frac{R - (y_{\text{center}} - y_{\text{contact}})}{R}\right) \)

• Manually select three points on the image

• Fit a circle using three points

• Contact angle derived geometrically:  

  \( \theta = 90^\circ - \arcsin\left(\frac{R - (y_{\text{center}} - y_{\text{contact}})}{R}\right) \)

✅ 定位 / Role：教学演示（Educational demo）

2️⃣ 2.2.py —— 手动点选 + 多项式拟合

• 鼠标点击左右两个接触点

• 基于灰度梯度提取液滴轮廓

• 使用二次多项式拟合边缘

• 计算切线与基线夹角

• Click left and right contact points manually

• Extract droplet contour using gray-level gradients

• Fit a second-order polynomial to the edge

• Compute the angle between the tangent and baseline

✅ 定位 / Role：轻量级手动测量（Lightweight manual measurement）

3️⃣ 3.7segmenter.py —— 液滴分割与基准线检测

• 输入增强后的 ROI

• 使用 Otsu 阈值 + 形态学操作分割液滴

• 根据轮廓宽度突变检测基准线（baseline）

• Input: enhanced ROI

• Segment droplet using Otsu thresholding + morphology

• Detect baseline from contour width collapse

✅ 定位 / Role：分割模块（Segmentation module）

4️⃣ 3.7YLCal.py —— Young-Laplace 拟合器

• 输入：轮廓、基准线、初始半径

• 数值求解 Young-Laplace 微分方程

• 使用最小二乘法拟合接触角

• Input: contour, baseline, initial radius

• Numerically solve the Young–Laplace ODE

• Fit contact angle using least-squares optimization

✅ 定位 / Role：数值计算模块（Numerical solver module）

5️⃣ 3.7picbetter.py —— 半自动主程序

• 串联分割器与拟合器

• 自动完成：定位 → 增强 → 分割 → 拟合

• Integrate segmenter and Young–Laplace calculator

• Pipeline: localization → enhancement → segmentation → fitting

✅ 定位 / Role：实验室级半自动流程（Semi-automatic lab pipeline）

6️⃣ 3.7app.py —— Web 在线测量系统

• 基于 Flask 的 Web 应用

• 用户上传图片 → 自动返回接触角与过程图

• 适合多人协作与远程使用

• Flask-based web application

• Upload image → automatically return contact angles and diagnostics

• Suitable for multi-user and remote scenarios

✅ 定位 / Role：可部署服务（Deployable service）

7️⃣ 4.4.py —— 工业级全自动工具

• 支持多尺度 Hough 圆检测

• 多方法互验：

  • Young-Laplace

  • 圆拟合

  • 多项式拟合

• 自动回退机制，鲁棒性强

• Multi-scale Hough circle detection

• Multi-method validation:

  • Young–Laplace

  • Circle fitting

  • Polynomial fitting

• Automatic fallback for robustness

✅ 定位 / Role：工程与科研通用方案（General-purpose research tool）

8️⃣ 1.7.py —— AI + 物理模型（最强方案）

• 使用 SAM（Segment Anything Model）进行 AI 分割

• 传统 CV（Sobel + RANSAC）检测基底基准线

• Young-Laplace 方程拟合 + 多项式回退

• 精度最高，但部署复杂

• AI-based droplet segmentation using SAM

• Baseline detection via Sobel + RANSAC

• Young–Laplace fitting with polynomial fallback

• Highest accuracy, but most complex to deploy

✅ 定位 / Role：研究级方案（Research-grade solution）


---
### 🚀 Usage | 使用方式

#### Method 1: Web App (Recommended) | 方式一：Web 应用（推荐）

Start the Flask server and access the web interface in your browser. All processing is fully automated — no manual clicks needed.

```bash
python AP_app.py
```

Then open your browser and visit: `http://127.0.0.1:5000`

启动 Flask 服务器，在浏览器中访问 Web 界面。全部流程自动完成，无需手动操作。

```bash
python AP_app.py
```

然后在浏览器中打开：`http://127.0.0.1:5000`

---

#### Method 2: Manual Test Tool | 方式二：手动测试工具

Run the test script and manually click two points on the droplet image for potentially better results.

```bash
python Test/TE_contact_angle.py
```

运行测试脚本，手动在液滴图像上选取两个特征点，可能获得更优结果。

```bash
python Test/TE_contact_angle.py
```

---

### 🛠️ Technology Stack | 技术栈

| Category | 
| Image Processing | OpenCV (Hough Circle Transform, CLAHE, Otsu thresholding, morphological operations) 
| OpenCV（霍夫圆变换、CLAHE、Otsu 阈值分割、形态学操作） |
| Numerical Computing | SciPy (`solve_ivp` for ODE, `least_squares` for non-linear optimization）
| SciPy（`solve_ivp` 求解常微分方程、`least_squares` 非线性最小二乘优化） |
| Web Framework | Flask 
| Flask |
| Visualization | Matplotlib 
| Matplotlib |
| Enhancement (WIP) | `Enhancer/freeenhancer.py` — planned integration with external APIs or local GPU 
| `Enhancer/freeenhancer.py` — 计划集成外部 API 或本地 GPU |

---

### ⚙️ Requirements | 环境依赖

```bash
pip install opencv-python scipy flask matplotlib numpy
```

Or save the following as `requirements.txt`:

```txt
opencv-python>=4.5.0
scipy>=1.7.0
flask>=2.0.0
matplotlib>=3.4.0
numpy>=1.21.0
```

---

### 📌 Notes | 注意事项

- This software is **for internal use at CUPET only**.
- 本程序 **仅供 CUPET 内部使用**。
- The `Enhancer/` module is under active development and subject to change.
- `Enhancer/` 模块正在开发中，接口可能变更。

---

### 👤 Author | 作者

| **Developer** | 8cheh |
| **AI Collaborator** | Qwen |
| **Affiliation** | CUPET |

---

*If you find this project helpful, please give it a ⭐!*
*如果这个项目对你有帮助，欢迎给个 ⭐！*
