"""
kalman_core.py —— 手写一维卡尔曼滤波核心类（从零写，不调库）

被 10_kalman1d.py（可视化 demo）和 11_kalman_track_box.py（跟踪框中心）共用。

矩阵写法（2x2），但变量只有 [位置, 速度]，所以你现在能手推，以后能直接扩成 4x4（CV/CA）。

五个核心式子（合上书默写版）：
    预测  x⁻ = A x                 P⁻ = A P Aᵀ + Q
    增益  K  = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
    更新  x  = x⁻ + K (z − H x⁻)    P = (I − K H) P⁻
"""
from __future__ import annotations

import numpy as np


class Kalman1D:
    """一维匀速（CV）模型：状态 x = [位置, 速度]^T，只观测位置。

    q : 加速度的方差，物理含义 = "我有多不相信'匀速'这个假设"。q 越大越信观测。
    r : 观测噪声方差（detector 抖动多大就填多大，单位跟位置一致）。
    """

    def __init__(self, x0: float = 0.0, v0: float = 0.0, dt: float = 1.0,
                 q: float = 1e-2, r: float = 1e-1, p0: float | tuple[float, float] = 1.0) -> None:
        self.x = np.array([[float(x0)], [float(v0)]])
        self.dt = float(dt)
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        # p0 给标量=位置/速度同方差；给 (位置方差, 速度方差)=分开设。
        # 跟踪里"速度先验"通常要设得很大（比如 300²），否则滤波器不敢改速度，会一直滞后。
        if np.isscalar(p0):
            self.P = np.eye(2) * float(p0)  # type: ignore[arg-type]
        else:
            p0v = tuple(p0)
            self.P = np.diag([float(p0v[0]), float(p0v[1])])
        self.R = np.array([[float(r)]])
        self.q = float(q)
        self.Q = self.make_Q(self.q, self.dt)
        self.last_K: np.ndarray | None = None

    @staticmethod
    def make_Q(q: float, dt: float) -> np.ndarray:
        """白噪声加速度模型的离散 Q。

        别用 np.eye(2)*q —— 那样位置和速度的噪声互不相干，物理上不成立，
        会让滤波器的速度和位置打架。这个 4 项结构来自对连续模型做积分。
        """
        return q * np.array([[dt**4 / 4, dt**3 / 2],
                             [dt**3 / 2, dt**2]])

    def predict(self) -> None:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z: float | None) -> tuple[float, float]:
        """z=None 表示这一帧没检测到目标 → 只预测不更新，也就是"丢帧跟踪"。"""
        if z is None:
            return self.x[0, 0], self.x[1, 0]

        y = np.array([[float(z)]]) - self.H @ self.x          # 新息 innovation
        S = self.H @ self.P @ self.H.T + self.R               # 新息协方差
        K = self.P @ self.H.T @ np.linalg.inv(S)              # 卡尔曼增益
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        self.last_K = K
        return float(self.x[0, 0]), float(self.x[1, 0])

    def step(self, z: float | None) -> tuple[float, float]:
        self.predict()
        return self.update(z)

    @property
    def pos(self) -> float:
        return float(self.x[0, 0])

    @property
    def vel(self) -> float:
        return float(self.x[1, 0])

    @property
    def gain(self) -> float:
        return float(self.last_K[0, 0]) if self.last_K is not None else float("nan")

    def mahalanobis(self, z: float) -> float:
        """新息的马氏距离 —— 判断"这个观测是不是离谱到不该信"（关联门限的数学依据）。"""
        y = np.array([[float(z)]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(np.sqrt(y.T @ np.linalg.inv(S) @ y))
