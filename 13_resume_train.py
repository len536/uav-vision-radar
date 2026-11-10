"""
13_resume_train.py —— 从断点续训（用 last.pt 恢复）

为什么需要它：
    后台任务有约 1 小时上限，50 轮训不完就会被杀。last.pt 里保存了
    权重 + 优化器状态 + 当前 epoch + 全部超参，用 resume=True 可以无缝接上，
    不会浪费之前跑过的轮次。

用法：
    python 13_resume_train.py                      # 接着上次的 full 训练跑
    python 13_resume_train.py --ckpt xxx/last.pt   # 指定别的断点
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/train/full/weights/last.pt")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = here / ckpt

    if not ckpt.is_file():
        raise SystemExit(f"找不到断点文件: {ckpt}")

    model = YOLO(str(ckpt))
    # resume=True 会读取 ckpt 里保存的 args，自动恢复 epoch / 优化器 / 数据增强状态
    model.train(resume=True)


if __name__ == "__main__":
    main()
