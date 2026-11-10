"""
11_kalman_track_box.py —— 把 YOLO 检测框中心喂给卡尔曼（视觉 → 滤波的桥）

这一步是"视觉"和"融合"之间真正的接缝：
    YOLO 每帧给一个框 → 取中心 (cx, cy) → 当作观测量 z → KF 输出平滑位置 + 速度
    KF 输出的速度，就是雷达那边要对比、要参与融合的量。

本脚本用合成轨迹 + 人为抖动 + 人为丢帧，模拟 YOLO 的输出（不依赖 torch，先跑通逻辑）。
真接 YOLO 时，把 simulate_detections() 换成读 results[0].boxes 的循环即可，接口完全一样。

用法：
    python 11_kalman_track_box.py
输出：
    fig/demo4_track_box.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from uavfusion.kalman1d import Kalman1D  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SIGMA_PX = 6.0      # 模拟检测框中心的抖动（真实 YOLO 大概 3~10 px，小目标更大）
DROP = (60, 70)     # 模拟这 10 帧漏检（遮挡 / conf 掉了）


def simulate_detections(n: int = 150, seed: int = 3):
    """返回 (t, dt, cx_true, cy_true, dets)，dets 里 None 表示这一帧没检到。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 5.0, n)        # 5 秒、30 fps
    dt = float(t[1] - t[0])
    cx_true = 300 + 120 * t                                     # 120 px/s 向右
    cy_true = 500 - 100 * t + 15 * np.sin(2 * np.pi * 0.5 * t)   # 向左上，带轻微摆动
    dets = []
    for i in range(n):
        if DROP[0] <= i < DROP[1]:
            dets.append(None)
        else:
            dets.append((cx_true[i] + rng.normal(0, SIGMA_PX),
                         cy_true[i] + rng.normal(0, SIGMA_PX)))
    return t, dt, cx_true, cy_true, dets


def main() -> None:
    out = Path("fig")
    out.mkdir(parents=True, exist_ok=True)
    t, dt, cx_true, cy_true, dets = simulate_detections()

    first = next(d for d in dets if d is not None)
    # q 的量纲是 px²/s⁴，数量级看着"离谱"是正常的：它约等于目标加速度标准差的平方。
    # 这个量级是从 1e3 扫到 1e7 试出来的，不是拍脑袋。
    kf_x = Kalman1D(first[0], 0.0, dt=dt, q=1e5, r=SIGMA_PX**2, p0=(SIGMA_PX**2, 300.0**2))
    kf_y = Kalman1D(first[1], 0.0, dt=dt, q=1e5, r=SIGMA_PX**2, p0=(SIGMA_PX**2, 300.0**2))

    est_x, est_y, raw_x, raw_y = [], [], [], []
    for z in dets:
        kf_x.predict()
        kf_y.predict()
        zx = None if z is None else z[0]
        zy = None if z is None else z[1]
        est_x.append(kf_x.update(zx)[0])
        est_y.append(kf_y.update(zy)[0])
        raw_x.append(np.nan if zx is None else zx)
        raw_y.append(np.nan if zy is None else zy)

    raw_x, raw_y = np.array(raw_x), np.array(raw_y)
    est_x, est_y = np.array(est_x), np.array(est_y)

    m = ~np.isnan(raw_x)
    rmse_raw = float(np.sqrt(np.mean((raw_x[m] - cx_true[m]) ** 2 + (raw_y[m] - cy_true[m]) ** 2)))
    rmse_kf = float(np.sqrt(np.mean((est_x - cx_true) ** 2 + (est_y - cy_true) ** 2)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(cx_true, cy_true, "k--", lw=1, label="真值")
    ax[0].plot(raw_x, raw_y, ".", ms=6, c="tab:gray", alpha=0.8, label="YOLO 框中心（抖动）")
    ax[0].plot(est_x, est_y, "-", lw=2, c="tab:red", label="KF 平滑后")
    ax[0].invert_yaxis()   # 图像坐标系 y 向下
    ax[0].set_xlabel("cx (px)")
    ax[0].set_ylabel("cy (px)")
    ax[0].set_title("图像平面轨迹")
    ax[0].legend(fontsize=8)

    ax[1].plot(t, cx_true, "k--", lw=1, label="真值 cx")
    ax[1].plot(t, raw_x, ".", ms=5, c="tab:gray", label="观测")
    ax[1].plot(t, est_x, lw=2, c="tab:red", label="KF 估计")
    ax[1].axvspan(t[DROP[0]], t[DROP[1] - 1], color="orange", alpha=0.25, label="漏检区间")
    ax[1].set_xlabel("时间 (s)")
    ax[1].set_title("漏检时靠预测硬撑")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    p = out / "demo4_track_box.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)

    print(f"RMSE(观测) {rmse_raw:.2f} px  ->  RMSE(滤波) {rmse_kf:.2f} px  "
          f"(改善 {100 * (1 - rmse_kf / rmse_raw):.1f}%)")
    print(f"末尾速度估计: vx={kf_x.vel:.1f} px/s, vy={kf_y.vel:.1f} px/s  "
          f"(真值 120.0 / -100.0 左右)")
    print(f"图: {p}")
    print("\n两个必须记住的点：")
    print("  1. 第 60~70 帧完全没有观测，KF 靠'匀速假设'外推 —— 雷达丢点、相机遮挡时系统靠的就是这个。")
    print("  2. 真实系统里 q 要靠实验调。调法：固定 R=测量抖动²，q 从 1e3 扫到 1e7，画 RMSE 取谷底。")


if __name__ == "__main__":
    main()
