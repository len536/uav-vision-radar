"""
02_verify_labels.py —— 把转换后的 YOLO 标签画回原图，肉眼核对

用途：标注转换是整条链路最容易错、又最难发现的一步。
      "框整体偏移"、"框变成细长条"、"cx/cy 和 x1/y1 搞混" 都只能靠看图发现。
      跑完打开 out_verify/ 里的图，框应该严丝合缝贴住目标。

用法：
    python 02_verify_labels.py --root datasets/VisDrone --split train --n 6

输出：out_verify/<split>_<序号>.jpg
"""
from __future__ import annotations

import argparse
import colorsys
import random
from pathlib import Path

import cv2

NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def color_of(cid: int) -> tuple[int, int, int]:
    h = (cid * 0.13) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/VisDrone")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=6, help="随机抽几张")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_verify")
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"没有图片: {img_dir.resolve()}")

    random.seed(args.seed)
    picks = random.sample(imgs, min(args.n, len(imgs)))

    for i, img_path in enumerate(picks, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print("读图失败:", img_path)
            continue
        H, W = img.shape[:2]

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        n_box = 0
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:])
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
                c = color_of(cid)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
                label = NAMES[cid] if cid < len(NAMES) else str(cid)
                cv2.putText(img, label, (x1, max(y1 - 2, 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
                n_box += 1

        dst = out_dir / f"{args.split}_{i:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(dst), img)
        print(f"[{i}] {img_path.name}: {n_box} 个框 -> {dst}")

    print("\n打开 out_verify/ 用看图软件逐张看：")
    print("  * 框应贴着目标边缘，不偏不斜           -> 通过")
    print("  * 框整体往左上/右下偏移               -> 归一化用错了宽高（w/h 写反）")
    print("  * 框变成扁条/竖条                     -> 格式写成了 x1 y1 x2 y2，应当是 cx cy w h")
    print("  * 框数量为 0 但图上明显有目标          -> 标签文件为空，检查 class 0 是否被全过滤")


if __name__ == "__main__":
    main()
