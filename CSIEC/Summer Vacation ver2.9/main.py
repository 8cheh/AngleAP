"""
命令行入口
支持单张图片和批量处理
"""
import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pipeline import ContactAnglePipeline


def visualize_result(result, save_path=None):
    """生成诊断可视化图"""
    if not result.get("contour") is not None:
        return

    roi = result.get("roi_image")
    if roi is None:
        return

    contour = result["contour"]
    baseline_y = result.get("baseline_y_roi", 0)
    left_pt = result.get("left_contact")
    right_pt = result.get("right_contact")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: 分割结果
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    axes[0].imshow(roi_rgb)
    if len(contour) > 0:
        axes[0].plot(contour[:, 0], contour[:, 1], "lime", linewidth=1.5, label="Contour")
    axes[0].axhline(y=baseline_y, color="red", linestyle="--", linewidth=2, label="Baseline")
    if left_pt is not None:
        axes[0].plot(left_pt[0], left_pt[1], "mo", markersize=10, label="Left contact")
    if right_pt is not None:
        axes[0].plot(right_pt[0], right_pt[1], "co", markersize=10, label="Right contact")
    axes[0].set_title(
        f"Segmentation ({result['steps'].get('segmentation', '?')})"
    )
    axes[0].legend(fontsize=8)

    # 右图: 角度信息
    axes[1].imshow(roi_rgb)
    axes[1].axhline(y=baseline_y, color="red", linestyle="--", linewidth=2)
    if len(contour) > 0:
        axes[1].plot(contour[:, 0], contour[:, 1], "lime", linewidth=1.5)
    axes[1].text(
        10, 20,
        f"Angle: {result['angle']:.1f}°\n"
        f"Left: {result['angle_left']:.1f}°  Right: {result['angle_right']:.1f}°\n"
        f"Method: {result['method']}  |  Confidence: {result['confidence_level']}",
        fontsize=10, color="white",
        bbox=dict(facecolor="black", alpha=0.7),
    )
    axes[1].set_title("Result")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def cmd_single(args):
    """处理单张图片"""
    pipeline = ContactAnglePipeline()
    result = pipeline.process(args.image)

    if not result["success"]:
        print(f"[ERROR] {result.get('error', 'Unknown error')}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  接触角测量结果")
    print(f"{'='*50}")
    print(f"  平均接触角:  {result['angle']:.2f}°")
    print(f"  左侧:        {result['angle_left']:.2f}°")
    print(f"  右侧:        {result['angle_right']:.2f}°")
    print(f"  算法:        {result['method']}")
    yl_str = f"{result['angle_yl']:.2f}°" if result.get('angle_yl') is not None else "N/A"
    local_str = f"{result['angle_local']:.2f}°" if result.get('angle_local') is not None else "N/A"
    print(f"  Y-L拟合:     {yl_str}")
    print(f"  局部拟合:    {local_str}")
    print(f"  置信度:      {result['confidence_level']} ({result['confidence_score']:.2f})")
    print(f"  耗时:        {result['metadata']['elapsed_ms']:.0f} ms")
    print(f"{'='*50}")
    print(f"\n管线详情:")
    for step_name, step_method in result["steps"].items():
        print(f"  {step_name}: {step_method}")

    if args.save or args.show:
        save_path = None
        if args.save:
            base = os.path.splitext(os.path.basename(args.image))[0]
            save_path = f"{base}_result.png"
        visualize_result(result, save_path)
        if save_path:
            print(f"\n可视化结果已保存到: {save_path}")


def cmd_batch(args):
    """批量处理"""
    import glob

    pipeline = ContactAnglePipeline()
    patterns = args.batch if isinstance(args.batch, list) else [args.batch]
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(pattern))
    image_files = sorted(set(image_files))

    if not image_files:
        print("[ERROR] 没有找到匹配的图片文件")
        sys.exit(1)

    print(f"找到 {len(image_files)} 张图片, 开始批量处理...\n")

    results = []
    for i, path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {os.path.basename(path)} ... ", end="", flush=True)
        r = pipeline.process(path)
        if r["success"]:
            print(f"{r['angle']:.1f}° ({r['method']}, {r['confidence_level']})")
            results.append({
                "file": os.path.basename(path),
                "path": path,
                "angle": r["angle"],
                "angle_left": r["angle_left"],
                "angle_right": r["angle_right"],
                "method": r["method"],
                "confidence": r["confidence_score"],
                "level": r["confidence_level"],
            })
        else:
            print(f"FAILED: {r.get('error', '?')}")
            results.append({
                "file": os.path.basename(path),
                "path": path,
                "angle": None,
                "error": r.get("error", "?"),
            })

    # 汇总
    success_count = sum(1 for r in results if r["angle"] is not None)
    print(f"\n{'='*50}")
    print(f"  批量处理完成: {success_count}/{len(results)} 成功")

    if success_count > 0:
        angles = [r["angle"] for r in results if r["angle"] is not None]
        print(f"  角度范围: {min(angles):.1f}° ~ {max(angles):.1f}°")
        print(f"  平均: {np.mean(angles):.1f}° ± {np.std(angles):.1f}°")

    # 导出 CSV
    if args.output:
        import csv
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file", "angle", "angle_left", "angle_right",
                    "method", "confidence", "level",
                ],
            )
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, "") for k in writer.fieldnames}
                writer.writerow(row)
        print(f"  结果已导出到: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="接触角自动测量工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 单张图片
    sp_single = subparsers.add_parser("single", help="处理单张图片")
    sp_single.add_argument("image", help="图片路径")
    sp_single.add_argument("--save", action="store_true", help="保存可视化结果")
    sp_single.add_argument("--show", action="store_true", help="显示可视化窗口")

    # 批量处理
    sp_batch = subparsers.add_parser("batch", help="批量处理图片")
    sp_batch.add_argument("batch", nargs="+", help="图片文件或通配符 (如 '*.jpg')")
    sp_batch.add_argument("-o", "--output", help="导出 CSV 文件路径")

    args = parser.parse_args()

    if args.command == "single":
        cmd_single(args)
    elif args.command == "batch":
        cmd_batch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
