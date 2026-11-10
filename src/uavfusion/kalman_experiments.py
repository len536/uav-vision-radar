"""
10_kalman1d.py —— 一维卡尔曼滤波可视化实验（核心类在 kalman_core.py，这个文件只负责"看"）

三个 demo：
    1) 静态目标 + 噪声观测   -> 看卡尔曼增益 K 从 ~1 收敛到 ~0（越测越信模型）
    2) 匀速直线 + 微机动      -> 出"真值 / 观测 / 估计"三线图（组会汇报最好用的图）
    3) Q、R 各放大 10 倍      -> 看清"信模型"还是"信观测"的物理含义

用法：
    python 10_kalman1d.py            # 三个全跑
    python 10_kalman1d.py --demo 2
输出：fig/demo1_static.png, fig/demo2_tracking.png, fig/demo3_qr.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面也能出图
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from uavfusion.kalman1d import Kalman1D  # noqa: E402

# 让图里的中文正常显示（Windows 自带雅黑）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


# =====================================================================
# Demo 1：静态目标，看增益 K 收敛
# =====================================================================
def demo_static(out: Path) -> None:
    rng = np.random.default_rng(0)
    n, truth, sigma = 50, 10.0, 2.0
    zs = truth + rng.normal(0, sigma, n)

    # q 极小 = 强烈相信"目标不动"，r 用观测噪声方差
    kf = Kalman1D(x0=0.0, v0=0.0, dt=1.0, q=1e-4, r=sigma**2, p0=100.0)
    est, gains = [], []
    for z in zs:
        kf.predict()
        est.append(kf.update(z)[0])
        gains.append(kf.gain)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].scatter(range(n), zs, s=12, c="tab:gray", label="观测 z")
    ax[0].plot(range(n), est, lw=2, c="tab:red", label="KF 估计")
    ax[0].axhline(truth, ls="--", c="k", lw=1, label="真值")
    ax[0].set_title("静态目标：估计快速收敛，之后几乎不动")
    ax[0].legend(fontsize=8)

    ax[1].plot(range(n), gains, lw=2, c="tab:blue")
    ax[1].set_title("卡尔曼增益 K：从 ~1 掉到 ~0（越来越信模型）")
    ax[1].set_xlabel("步数")
    fig.tight_layout()
    p = out / "demo1_static.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)

    print(f"[demo1] 后 25 步估计均值 {np.mean(est[25:]):.3f}  真值 {truth:.3f}  "
          f"末段 K={gains[-1]:.3f}  -> {p}")
    print("        注意：单步估计会随观测波动，但均值和方差都在收敛 —— 这才是 KF 的价值。")


# =====================================================================
# Demo 2：匀速直线 + 微机动，真值 / 观测 / 估计 三线图
# =====================================================================
def demo_tracking(out: Path) -> None:
    rng = np.random.default_rng(1)
    dt, n = 0.1, 200
    t = np.arange(n) * dt
    # 5 m/s 匀速主线，叠加 0.5 m 幅度、0.2 Hz 的缓慢机动（无人机悬停/微调）
    truth_p = 5.0 * t + 0.5 * np.sin(2 * np.pi * 0.2 * t)
    sigma = 0.6
    zs = truth_p + rng.normal(0, sigma, n)

    kf = Kalman1D(x0=zs[0], v0=0.0, dt=dt, q=10.0, r=sigma**2, p0=1.0)
    est_p, P_pos = [], []
    for z in zs:
        kf.predict()
        p, _ = kf.update(z)
        est_p.append(p)
        P_pos.append(kf.P[0, 0])

    est_p = np.array(est_p)
    rmse_raw = float(np.sqrt(np.mean((zs - truth_p) ** 2)))
    rmse_kf = float(np.sqrt(np.mean((est_p - truth_p) ** 2)))

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(t, truth_p, "k--", lw=1.2, label="真值")
    ax[0].scatter(t, zs, s=9, c="tab:gray", alpha=0.7, label=f"观测 (RMSE={rmse_raw:.2f})")
    ax[0].plot(t, est_p, lw=2.2, c="tab:red", label=f"KF 估计 (RMSE={rmse_kf:.2f})")
    ax[0].set_ylabel("位置 (m)")
    ax[0].legend(fontsize=9)
    ax[0].set_title("一维卡尔曼滤波：真值 / 观测 / 估计")

    ax[1].plot(t, P_pos, lw=1.8, c="tab:blue")
    ax[1].set_ylabel("P[0,0]")
    ax[1].set_xlabel("时间 (s)   ——   P 收敛到稳态 = 位置估计的方差稳定")
    fig.tight_layout()
    p = out / "demo2_tracking.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)

    print(f"[demo2] RMSE 观测 {rmse_raw:.3f} -> 滤波 {rmse_kf:.3f} "
          f"(改善 {100 * (1 - rmse_kf / rmse_raw):.1f}%)  -> {p}")
    print("        试验一下：把真值里的正弦改成 2 m / 0.8 Hz（急机动），RMSE 会反而变差。")
    print("        原因：CV 模型假设加速度为 0，机动太猛时模型本身就是错的 —— 这就是 CA 模型 / IMM 的由来。")


# =====================================================================
# Demo 3：Q / R 各放大 10 倍，看曲线性格变化
# =====================================================================
def demo_qr(out: Path) -> None:
    rng = np.random.default_rng(2)
    dt, n = 0.1, 150
    t = np.arange(n) * dt
    truth_p = 3.0 * t + 0.5 * np.sin(2 * np.pi * 0.2 * t)
    sigma = 0.6
    zs = truth_p + rng.normal(0, sigma, n)

    settings = [
        ("基准 q=10, R=σ²", 10.0, sigma**2),
        ("Q x10 → 更信观测（跟得快、更抖）", 100.0, sigma**2),
        ("R x10 → 更信模型（更平滑、会滞后）", 10.0, (sigma * 10) ** 2),
        ("Q /100 → 极信模型（几乎成直线）", 0.1, sigma**2),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True, sharey=True)
    for ax, (title, q, r) in zip(axs.ravel(), settings):
        kf = Kalman1D(x0=zs[0], v0=0.0, dt=dt, q=q, r=r, p0=1.0)
        est = []
        for z in zs:
            kf.predict()
            est.append(kf.update(z)[0])
        ax.plot(t, truth_p, "k--", lw=1, label="真值")
        ax.scatter(t, zs, s=7, c="tab:gray", alpha=0.6)
        ax.plot(t, est, lw=2, c="tab:red")
        ax.set_title(title, fontsize=9)
    fig.suptitle("Q = 你多不信'匀速假设'   |   R = 你多不信'传感器读数'", fontsize=11)
    fig.tight_layout()
    p = out / "demo3_qr.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"[demo3] 四组 Q/R 对照 -> {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=0, choices=[0, 1, 2, 3], help="0=全部")
    ap.add_argument("--out", default="fig")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.demo in (0, 1):
        demo_static(out)
    if args.demo in (0, 2):
        demo_tracking(out)
    if args.demo in (0, 3):
        demo_qr(out)

    print("\n看完图问自己三个问题（答不出来说明还没懂）：")
    print("  1. 为什么一开始 K 接近 1，后来接近 0？")
    print("  2. Q 变大曲线为什么更抖？R 变大为什么更迟钝、会滞后？")
    print("  3. P 收敛到稳态值，物理含义是「我对位置的把握稳定在 ±多少米」？")


if __name__ == "__main__":
    main()
