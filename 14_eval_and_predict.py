"""
14_eval_and_predict.py —— 训练完的验收脚本（跑完 50 轮后用这个）

做三件事：
  1. 在验证集上算指标（mAP50 / mAP50-95 / 逐类 AP）
  2. 抽 N 张验证集图跑推理，把检测结果画出来存到 out_predict/
  3. 打印一份"人话版"结论：哪些类能用，哪些类基本废

用法：
    python 14_eval_and_predict.py                                  # 用 full 训练的 best.pt
    python 14_eval_and_predict.py --weights runs/train/smoke/weights/best.pt --n 4
    python 14_eval_and_predict.py --conf 0.35                      # 调置信度阈值看效果变化
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

DATA = "visdrone_local.yaml"
VAL_IMAGES = (
    "C:/Users/ASUS/.cache/kagglehub/datasets/banuprasadb/visdrone-dataset/"
    "versions/1/VisDrone_Dataset/VisDrone2019-DET-val/images"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/train/full/weights/best.pt")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--n", type=int, default=4, help="抽几张图做可视化")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    w = Path(args.weights)
    if not w.is_absolute():
        w = here / w
    if not w.is_file():
        raise SystemExit(f"找不到权重: {w}\n（训练还没跑完？先跑 继续训练.bat）")

    print(f"权重: {w}\n")

    # ---------- 1. 验证集指标 ----------
    model = YOLO(str(w))
    print("=" * 60)
    print("验证集指标")
    print("=" * 60)
    metrics = model.val(
        data=str(here / args.data),
        imgsz=args.imgsz,
        split="val",
        conf=0.001,      # 算 mAP 要用低阈值，别用 0.25
        iou=0.6,
        plots=True,
    )

    names = model.names
    ap50 = metrics.box.ap50          # 逐类 AP@0.5
    print(f"\n总体: mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    print(f"      P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}\n")

    print(f"{'类别':<18}{'AP@50':>8}   评价")
    print("-" * 60)
    for i, name in enumerate(names.values() if isinstance(names, dict) else names):
        v = float(ap50[i])
        if v >= 0.5:
            tag = "好用，可直接进项目"
        elif v >= 0.3:
            tag = "勉强可用，需调阈值"
        elif v >= 0.1:
            tag = "很弱，只能当参考"
        else:
            tag = "基本失效（小目标/样本少）"
        print(f"{name:<18}{v:>8.3f}   {tag}")

    # ---------- 2. 抽图可视化 ----------
    out = here / "out_predict"
    out.mkdir(exist_ok=True)
    imgs = sorted(Path(VAL_IMAGES).glob("*.jpg"))[: args.n]
    if not imgs:
        print(f"\n验证图目录为空: {VAL_IMAGES}")
        return

    print(f"\n{'=' * 60}\n可视化 {len(imgs)} 张（conf>={args.conf}）\n{'=' * 60}")
    for i, p in enumerate(imgs, 1):
        res = model.predict(str(p), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dst = out / f"pred_{i:02d}_{p.stem}.jpg"
        cv2.imwrite(str(dst), res.plot())      # res.plot() 自带画框+标签
        print(f"[{i}] {p.name}: {len(res.boxes)} 个框 -> {dst.name}")

    print(f"\n打开 {out} 看效果。重点看：")
    print("  * 大目标（车、公交）框得住吗")
    print("  * 密集小目标（远处行人）漏多少")
    print("  * 有没有把 car 误判成 van/truck（VisDrone 这俩类本来就难分）")


if __name__ == "__main__":
    main()
