"""
01_visdrone2yolo.py —— 把 VisDrone2019-DET 的标注转成 YOLO 格式

VisDrone 原始标注行（逗号分隔，8 个字段）：
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

    * 注意：静态图片集（DET）里没有 frame_id，第 1 个字段就是 bbox_left。
      只有视频集（VID）才有 frame_id。网上很多教程在这里抄错，转换完框会整体偏移。
    * category == 0 表示 "ignored regions"（忽略区域，通常是大片背景），必须剔除；
      否则模型会被迫去学 "这一大片背景是一个目标"，mAP 会莫名其妙地低。
    * score == 0 的行多半也是无效/忽略标注，一并丢掉。

YOLO 目标格式（空格分隔，全部归一化到 0~1）：
    class_id  cx  cy  w  h

用法：
    # 数据集解压到 datasets/VisDrone 下，保留原来的 VisDrone2019-DET-xxx 目录
    python 01_visdrone2yolo.py --root datasets/VisDrone --out datasets/VisDrone

    # 转换完再核对框（强烈建议，别跳）
    python 02_verify_labels.py --root datasets/VisDrone --split train --n 6
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

# VisDrone 原始 category -> YOLO class_id
CLASS_MAP: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

SPLITS = ["train", "val", "test-dev", "test-challenge"]


def convert_split(root: Path, split: str, out_root: Path, min_wh: float, copy_images: bool) -> None:
    src = root / f"VisDrone2019-DET-{split}"
    ann_dir = src / "annotations"
    img_dir = src / "images"

    if not img_dir.is_dir():
        print(f"[skip] {split}: 找不到 {img_dir}")
        return
    if not ann_dir.is_dir():
        print(f"[skip] {split}: 没有 annotations（test-dev 本来就不给标注），只整理图片")
    else:
        print(f"[{split}] 图片 {len(list(img_dir.glob('*.jpg')))} 张，标注 {len(list(ann_dir.glob('*.txt')))} 个")

    # 目标目录
    dst_img = out_root / "images" / split.replace("-dev", "").replace("-challenge", "")
    lbl_dir = out_root / "labels" / split.replace("-dev", "").replace("-challenge", "")
    lbl_dir.mkdir(parents=True, exist_ok=True)

    if copy_images:
        dst_img.mkdir(parents=True, exist_ok=True)

    if not ann_dir.is_dir():
        return

    cls_counter: Counter[int] = Counter()
    n_box = n_skip_ignore = n_skip_bad = 0

    for f in sorted(ann_dir.glob("*.txt")):
        img_path = img_dir / f"{f.stem}.jpg"
        if not img_path.is_file():
            n_skip_bad += 1
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        out_lines: list[str] = []
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                x1, y1, w, h = (float(v) for v in parts[:4])
                score = float(parts[4])
                cat = int(float(parts[5]))
            except ValueError:
                continue

            if cat not in CLASS_MAP or score <= 0:  # 0 = ignored region
                n_skip_ignore += 1
                continue
            if w < min_wh or h < min_wh:            # 过滤极端小框（默认不过滤）
                n_skip_ignore += 1
                continue

            # 归一化 + 裁剪到 [0,1]（VisDrone 有越界框）
            nx, ny = x1 / W, y1 / H
            nw, nh = w / W, h / H
            nx = min(max(nx, 0.0), 1.0)
            ny = min(max(ny, 0.0), 1.0)
            nw = min(nw, 1.0 - nx)
            nh = min(nh, 1.0 - ny)
            if nw <= 1e-6 or nh <= 1e-6:
                n_skip_ignore += 1
                continue

            cx, cy = nx + nw / 2, ny + nh / 2
            cid = CLASS_MAP[cat]
            cls_counter[cid] += 1
            n_box += 1
            out_lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # 空标签文件也要写（YOLO 允许负样本，但这里空文件意味着"图上确实没目标"）
        (lbl_dir / f"{f.stem}.txt").write_text("\n".join(out_lines), encoding="utf-8")

        if copy_images:
            shutil.copy2(img_path, dst_img / img_path.name)

    print(f"  -> {lbl_dir} : {n_box} 个框，剔除 {n_skip_ignore} 条（ignore/无效），缺图 {n_skip_bad}")
    if cls_counter:
        dist = "  ".join(f"{NAMES[k]}:{cls_counter[k]}" for k in sorted(cls_counter))
        print(f"  -> 类别分布: {dist}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/VisDrone", help="解压后的 VisDrone 根目录")
    ap.add_argument("--out", default=None, help="输出根目录，默认与 --root 相同")
    ap.add_argument("--splits", default="train,val", help="逗号分隔，默认 train,val")
    ap.add_argument("--min-wh", type=float, default=1.0, help="小于该像素宽/高的框丢弃（默认 1.0，几乎不过滤）")
    ap.add_argument("--copy-images", action="store_true",
                    help="把图片复制到 <out>/images/<split>/（YOLO 能直接读原始目录时可不开）")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) if args.out else root
    if not root.is_dir():
        raise SystemExit(f"数据集目录不存在: {root.resolve()}")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        convert_split(root, split, out_root, args.min_wh, args.copy_images)

    print("\n完成。下一步务必跑 02_verify_labels.py 肉眼核对框位置。")


if __name__ == "__main__":
    main()
