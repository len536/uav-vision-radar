"""
12_train_visdrone.py —— VisDrone 训练入口（Python API 版，绕开 CLI 的坑）

用法：
    python 12_train_visdrone.py                # 默认冒烟：3 epoch / 320
    python 12_train_visdrone.py --epochs 50 --imgsz 640 --batch 8 --name full

为什么不用 `yolo train ...` 命令行：
    Windows 下 CLI 的参数解析偶尔静默失败（本机实测 exit 1 无输出），
    用 API 写死参数更稳，也方便你在脚本里加自定义回调（比如训练中途存验证图）。

显存参考（RTX 5060 8GB）：
    imgsz=640 batch=8  -> 约 6.5GB，安全
    imgsz=640 batch=16 -> 容易 OOM
    imgsz=1280 batch=2 -> 约 7GB，勉强
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--data", default="configs/visdrone_local.yaml")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="smoke")
    ap.add_argument("--workers", type=int, default=0, help="Windows 上建议 0，避免多进程卡死")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    model = YOLO(str(here / args.model) if (here / args.model).is_file() else args.model)

    model.train(
        data=str(here / args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(here / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        # 这几项对 VisDrone 这种小目标密集场景有用：
        cache=False,          # 16GB 内存别开 cache，会吃满
        patience=0,           # 冒烟不早停；正式训练可设成 15
        plots=True,           # 出 PR 曲线、混淆矩阵，组会能直接用
        val=True,
        seed=0,
        deterministic=False,
    )


if __name__ == "__main__":
    main()
