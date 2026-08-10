"""
接触角测量 Web 应用
阶段1: 上传 → 自动检测液滴 → 显示 ROI + 轮廓
阶段2: 用户点击左右接触点 → 双方法计算 → 显示拟合曲线
"""
import os, io, base64, traceback, uuid, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

from pipeline import ContactAnglePipeline

app = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
pipeline = ContactAnglePipeline()
SESSION = {}  # session_id → image_path


def _to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    return obj


def _b64(img_bgr):
    """BGR numpy → base64 PNG"""
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode()


def _make_result_fig(roi, contour, baseline_y, left_pt, right_pt, curves, ang):
    """生成结果图: 轮廓 + 基线 + 双拟合曲线 + 切线"""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # 轮廓
    if len(contour) > 0:
        ax.plot(contour[:, 0], contour[:, 1], "lime", lw=1.5, label="Contour")

    # 基线 + 接触点
    ax.axhline(y=baseline_y, color="red", ls="--", lw=2, label="Baseline")
    if left_pt:
        ax.plot(left_pt[0], left_pt[1], "mo", ms=12, label="Left contact")
        ax.plot(left_pt[0], left_pt[1], "mo", ms=4, color="white")
    if right_pt:
        ax.plot(right_pt[0], right_pt[1], "co", ms=12, label="Right contact")
        ax.plot(right_pt[0], right_pt[1], "co", ms=4, color="white")

    # 顶点坐标 (用于曲线坐标转换)
    zs = baseline_y - contour[:, 1]
    apex_idx = np.argmax(zs)
    x_apex = contour[apex_idx, 0]

    # Y-L 曲线 (黄色)
    if curves and curves.get("yl_left"):
        xc, zc = curves["yl_left"]
        # zc 是在 (x, z) 坐标系中 (z 向下为正), 转回 ROI y 坐标
        yl_y_left = baseline_y - zc
        yl_x_left = x_apex - xc
        ax.plot(yl_x_left, yl_y_left, "yellow", lw=2, label="Y-L fit")

    if curves and curves.get("yl_right"):
        xc, zc = curves["yl_right"]
        yl_y_right = baseline_y - zc
        yl_x_right = x_apex + xc
        ax.plot(yl_x_right, yl_y_right, "yellow", lw=2)

    # 局部多项式曲线 (橙色)
    if curves and curves.get("poly_left"):
        xf, zf = curves["poly_left"]
        py_y_left = baseline_y - zf
        py_x_left = x_apex - xf
        ax.plot(py_x_left, py_y_left, "orange", lw=2, label="Local poly")

    if curves and curves.get("poly_right"):
        xf, zf = curves["poly_right"]
        py_y_right = baseline_y - zf
        py_x_right = x_apex + xf
        ax.plot(py_x_right, py_y_right, "orange", lw=2)

    # 文字标注
    lines = [f"Average: {ang['angle_avg']:.1f}°"]
    if ang.get("angle_yl"):
        lines.append(f"Y-L: {ang['angle_yl']:.1f}°")
    if ang.get("angle_local"):
        lines.append(f"Local: {ang['angle_local']:.1f}°")
    lines.append(f"L: {ang['angle_left']:.1f}°  R: {ang['angle_right']:.1f}°")
    ax.text(5, roi.shape[0] - 5, "\n".join(lines),
            fontsize=9, color="white", va="bottom",
            bbox=dict(facecolor="black", alpha=0.7))

    ax.set_title(f"Result — {ang['method']}")
    ax.legend(fontsize=7, loc="upper right")
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ==================== API ====================

@app.route("/")
def index():
    return PAGE_HTML


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """阶段1: 上传图片 → 检测液滴 → 返回 ROI + 轮廓"""
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "请上传图片"})

    f = request.files["image"]
    sid = uuid.uuid4().hex
    fpath = os.path.join(UPLOAD_DIR, f"{sid}.jpg")
    f.save(fpath)

    r = pipeline.detect(fpath)
    if r.get("error"):
        os.remove(fpath)
        return jsonify({"ok": False, "error": r["error"]})

    SESSION[sid] = fpath

    return jsonify(_to_native({
        "ok": True,
        "session_id": sid,
        "roi_b64": _b64(r["roi"]),
        "contour": r["contour"],       # [[x,y], ...]
        "roi_w": r["roi"].shape[1],
        "roi_h": r["roi"].shape[0],
    }))


@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    """阶段2: 用户选定接触点 → 计算"""
    data = request.get_json()
    sid = data.get("session_id")
    left = data.get("left_contact")   # [x, y] or None
    right = data.get("right_contact") # [x, y] or None

    if not sid or sid not in SESSION:
        return jsonify({"ok": False, "error": "会话过期, 请重新上传"})

    fpath = SESSION[sid]
    r = pipeline.calculate(fpath,
                           tuple(left) if left else None,
                           tuple(right) if right else None)

    if r.get("error"):
        return jsonify({"ok": False, "error": r["error"]})

    roi = pipeline._stage1_cache.get(fpath, {}).get("roi")
    contour = roi is not None and np.array(r.get("contour", []))

    fig_b64 = ""
    if roi is not None:
        cached = pipeline._stage1_cache[fpath]
        fig_b64 = _make_result_fig(
            roi,
            cached["contour"],
            r["baseline_y"],
            r.get("left_contact"),
            r.get("right_contact"),
            r.get("curves"),
            {"angle_avg": r["angle"], "angle_yl": r["angle_yl"],
             "angle_local": r["angle_local"], "angle_left": r["angle_left"],
             "angle_right": r["angle_right"], "method": r["method"]},
        )

    return jsonify(_to_native({
        "ok": True,
        "angle": round(r["angle"], 2),
        "angle_yl": round(r["angle_yl"], 2) if r["angle_yl"] else None,
        "angle_local": round(r["angle_local"], 2) if r["angle_local"] else None,
        "angle_left": round(r["angle_left"], 2),
        "angle_right": round(r["angle_right"], 2),
        "method": r["method"],
        "confidence": r["confidence"],
        "elapsed_ms": r["elapsed_ms"],
        "baseline_y": r["baseline_y"],
        "left_contact": r.get("left_contact"),
        "right_contact": r.get("right_contact"),
        "result_b64": fig_b64,
    }))


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>接触角测量 — 交互选点</title>
<style>
:root {
  --bg:#f0f2f5; --card:#fff; --pri:#2563eb; --pri2:#1d4ed8;
  --ok:#10b981; --err:#ef4444; --txt:#1e293b; --t2:#64748b;
  --bd:#e2e8f0; --r:12px; --sh:0 1px 3px rgba(0,0,0,.08);
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--txt);min-height:100vh}
.header{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;padding:24px;text-align:center}
.header h1{font-size:1.5rem;font-weight:700}
.header p{margin-top:4px;opacity:.85;font-size:.85rem}
.container{max-width:900px;margin:0 auto;padding:28px 20px 60px}

.upload-zone{background:var(--card);border:2px dashed var(--bd);border-radius:var(--r);padding:36px 20px;text-align:center;cursor:pointer;transition:.2s;box-shadow:var(--sh)}
.upload-zone:hover,.upload-zone.drag{border-color:var(--pri);background:#eff6ff}
.upload-zone.done{border-style:solid;border-color:var(--ok);padding:16px}
.upload-zone input{display:none}
.upload-icon{width:48px;height:48px;margin:0 auto 12px;background:#eff6ff;border-radius:50%;display:flex;align-items:center;justify-content:center}
.hint{color:var(--t2);font-size:.8rem}

.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 22px;border-radius:8px;font-size:.9rem;font-weight:600;border:none;cursor:pointer;transition:.15s}
.btn-pri{background:var(--pri);color:#fff}.btn-pri:hover{background:var(--pri2)}
.btn-pri:disabled{background:#94a3b8;cursor:not-allowed}
.btn-out{background:#fff;color:var(--pri);border:1.5px solid var(--pri)}.btn-out:hover{background:#eff6ff}
.btn-act{margin-top:16px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

.selection-card{background:var(--card);border-radius:var(--r);box-shadow:0 10px 30px rgba(0,0,0,.1);margin-top:20px;overflow:hidden;display:none}
.selection-card.show{display:block}
.sel-header{background:#1e3a5f;color:#fff;padding:12px 18px;font-weight:600;font-size:.9rem;display:flex;justify-content:space-between;align-items:center}
.sel-body{padding:16px;text-align:center}
.sel-body canvas{max-width:100%;border-radius:8px;cursor:crosshair;box-shadow:var(--sh)}
.sel-info{margin-top:10px;font-size:.85rem;color:var(--t2)}

.result-card{background:var(--card);border-radius:var(--r);box-shadow:0 10px 30px rgba(0,0,0,.1);margin-top:20px;overflow:hidden;display:none}
.result-card.show{display:block}
.res-header{padding:18px;text-align:center;color:#fff}
.res-header.ok{background:linear-gradient(135deg,#10b981,#059669)}
.res-header.err{background:linear-gradient(135deg,#ef4444,#dc2626)}
.big{font-size:3rem;font-weight:800;line-height:1}
.big .unit{font-size:1.1rem;opacity:.8}
.res-detail{display:flex;gap:20px;justify-content:center;margin-top:6px;font-size:.85rem;opacity:.9}
.res-body{padding:16px}
.res-body img{width:100%;border-radius:8px;box-shadow:var(--sh);margin-top:8px}
.res-meta{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;font-size:.8rem;margin-top:10px}
.res-meta dt{color:var(--t2)}.res-meta dd{font-weight:500}
.badge{display:inline-block;padding:1px 8px;border-radius:20px;font-size:.7rem;font-weight:600}
.b-hi{background:#d1fae5;color:#065f46}.b-md{background:#fef3c7;color:#92400e}.b-lo{background:#fee2e2;color:#991b1b}

@media(max-width:600px){.big{font-size:2.2rem}.res-detail{flex-direction:column;gap:2px}}
</style>
</head>
<body>
<div class="header"><h1>接触角测量 — 交互选点</h1><p>上传图片 → 点击左右接触点 → 双方法计算</p></div>
<div class="container">

<div class="upload-zone" id="zone">
  <div id="prompt">
    <div class="upload-icon"><svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/></svg></div>
    <h3>点击上传或拖拽图片</h3><p class="hint">JPG / PNG / BMP</p>
  </div>
  <input type="file" id="fi" accept="image/*">
</div>

<div class="selection-card" id="selCard">
  <div class="sel-header"><span>点击左右两个接触点</span><span id="selStatus">请点击左侧接触点</span></div>
  <div class="sel-body"><canvas id="canvas"></canvas></div>
  <div class="sel-info" id="selInfo"></div>
  <div class="btn-act">
    <button class="btn btn-pri" id="btnCalc" disabled>计算接触角</button>
    <button class="btn btn-out" id="btnClear">清除选点</button>
    <button class="btn btn-out" id="btnRedo">重新上传</button>
  </div>
</div>

<div class="result-card" id="resCard"></div>
</div>

<script>
let sid = null, roiUrl = null, contour = [], roiW = 0, roiH = 0;
let leftPt = null, rightPt = null;

const zone = document.getElementById('zone'), fi = document.getElementById('fi');
const prompt = document.getElementById('prompt');
const selCard = document.getElementById('selCard'), selStatus = document.getElementById('selStatus');
const selInfo = document.getElementById('selInfo');
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const btnCalc = document.getElementById('btnCalc'), btnClear = document.getElementById('btnClear');
const btnRedo = document.getElementById('btnRedo');
const resCard = document.getElementById('resCard');

zone.addEventListener('click', () => fi.click());
fi.addEventListener('change', e => { if (e.target.files[0]) upload(e.target.files[0]); });
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag'); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });

async function upload(file) {
  if (!file.type.match(/image\/(jpeg|png|bmp|webp)/)) { alert('请上传 JPG/PNG/BMP'); return; }
  zone.classList.add('done');
  prompt.innerHTML = '<span class="spinner" style="border-top-color:#2563eb;border-color:#e2e8f0"></span> 检测液滴中...';
  const fd = new FormData(); fd.append('image', file);
  try {
    const r = await fetch('/api/detect', { method: 'POST', body: fd });
    const d = await r.json();
    if (!d.ok) { prompt.innerHTML = '<span style="color:#ef4444">'+d.error+'</span>'; zone.classList.remove('done'); return; }
    sid = d.session_id; contour = d.contour; roiW = d.roi_w; roiH = d.roi_h;
    roiUrl = 'data:image/png;base64,' + d.roi_b64;
    leftPt = null; rightPt = null; resCard.classList.remove('show'); resCard.innerHTML = '';
    drawCanvas();
    selCard.classList.add('show');
    prompt.innerHTML = '<span style="color:#10b981">液滴已检测</span>';
  } catch(e) { prompt.innerHTML = '<span style="color:#ef4444">网络错误</span>'; zone.classList.remove('done'); }
}

function drawCanvas() {
  const img = new Image();
  img.onload = () => {
    canvas.width = roiW; canvas.height = roiH;
    ctx.clearRect(0, 0, roiW, roiH);
    ctx.drawImage(img, 0, 0);
    // 画轮廓
    if (contour.length > 0) {
      ctx.beginPath();
      ctx.moveTo(contour[0][0], contour[0][1]);
      for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1]);
      ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 1.5; ctx.stroke();
    }
    // 画已选点
    if (leftPt) { drawMarker(leftPt[0], leftPt[1], '#ff00ff'); }
    if (rightPt) { drawMarker(rightPt[0], rightPt[1], '#00ffff'); }
    // 两点连线
    if (leftPt && rightPt) {
      ctx.beginPath(); ctx.moveTo(leftPt[0], leftPt[1]); ctx.lineTo(rightPt[0], rightPt[1]);
      ctx.strokeStyle = 'red'; ctx.setLineDash([6, 4]); ctx.lineWidth = 2; ctx.stroke();
      ctx.setLineDash([]);
    }
  };
  img.src = roiUrl;
}

function drawMarker(x, y, color) {
  ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
}

canvas.addEventListener('click', e => {
  if (!roiUrl) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = roiW / rect.width;
  const scaleY = roiH / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);
  if (x < 0 || y < 0 || x >= roiW || y >= roiH) return;

  if (!leftPt) { leftPt = [x, y]; selStatus.textContent = '请点击右侧接触点'; }
  else if (!rightPt) { rightPt = [x, y]; selStatus.textContent = '已选两点, 点击计算'; btnCalc.disabled = false; }
  else { leftPt = null; rightPt = null; selStatus.textContent = '请点击左侧接触点'; btnCalc.disabled = true; }
  drawCanvas();
  if (leftPt && rightPt) selInfo.textContent = `左: (${leftPt[0]}, ${leftPt[1]})  右: (${rightPt[0]}, ${rightPt[1]})`;
  else if (leftPt) selInfo.textContent = `左: (${leftPt[0]}, ${leftPt[1]})  —  请点右侧`;
  else selInfo.textContent = '';
});

btnClear.addEventListener('click', () => { leftPt = null; rightPt = null; selStatus.textContent = '请点击左侧接触点'; btnCalc.disabled = true; selInfo.textContent = ''; drawCanvas(); });
btnRedo.addEventListener('click', () => { sid = null; roiUrl = null; contour = []; leftPt = null; rightPt = null; selCard.classList.remove('show'); resCard.classList.remove('show'); resCard.innerHTML = ''; zone.classList.remove('done'); prompt.innerHTML = '<div class="upload-icon"><svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/></svg></div><h3>点击上传或拖拽图片</h3><p class="hint">JPG / PNG / BMP</p>'; });

btnCalc.addEventListener('click', async () => {
  if (!leftPt || !rightPt || !sid) return;
  btnCalc.disabled = true;
  btnCalc.innerHTML = '<span class="spinner"></span> 计算中...';
  try {
    const r = await fetch('/api/calculate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sid, left_contact: leftPt, right_contact: rightPt }),
    });
    const d = await r.json();
    if (!d.ok) { resCard.innerHTML = `<div class="result-card show"><div class="res-header err"><div>${d.error}</div></div></div>`; resCard.classList.add('show'); btnCalc.disabled = false; btnCalc.innerHTML = '计算接触角'; return; }
    const lv = d.confidence > 0.7 ? 'b-hi' : d.confidence > 0.4 ? 'b-md' : 'b-lo';
    const cl = d.confidence > 0.7 ? 'high' : d.confidence > 0.4 ? 'medium' : 'low';
    let ylStr = d.angle_yl !== null ? `Y-L: ${d.angle_yl}°` : '';
    let loStr = d.angle_local !== null ? `Local: ${d.angle_local}°` : '';
    resCard.innerHTML = `
      <div class="result-card show">
        <div class="res-header ok">
          <div class="big">${d.angle}<span class="unit">°</span></div>
          <div class="res-detail"><span>左: ${d.angle_left}°</span><span>右: ${d.angle_right}°</span></div>
          <div style="margin-top:4px"><span class="badge ${lv}">${cl}</span> ${d.method} · ${d.elapsed_ms}ms</div>
          ${ylStr||loStr ? `<div style="margin-top:2px;font-size:.8rem;opacity:.8">${ylStr}  ${loStr}</div>` : ''}
        </div>
        <div class="res-body">
          ${d.result_b64 ? '<img src="data:image/png;base64,'+d.result_b64+'">' : ''}
          <dl class="res-meta">
            <dt>基线Y</dt><dd>${d.baseline_y?.toFixed(1)}</dd>
            <dt>左接触点</dt><dd>(${d.left_contact?.[0]}, ${d.left_contact?.[1]})</dd>
            <dt>右接触点</dt><dd>(${d.right_contact?.[0]}, ${d.right_contact?.[1]})</dd>
            <dt>耗时</dt><dd>${d.elapsed_ms} ms</dd>
          </dl>
        </div>
      </div>`;
    resCard.classList.add('show');
  } catch(e) {
    resCard.innerHTML = `<div class="result-card show"><div class="res-header err"><div>网络错误: ${e.message}</div></div></div>`;
    resCard.classList.add('show');
  } finally {
    btnCalc.disabled = false;
    btnCalc.innerHTML = '计算接触角';
  }
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("=" * 50)
    print("  接触角测量 — 交互选点模式")
    print("  http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
