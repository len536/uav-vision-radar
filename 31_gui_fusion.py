"""
31_gui_fusion.py —— 三线程融合 Demo 骨架（W6~W7 目标，进组 demo 的原型）

架构（和 README 第 5 节的图完全对应）：

    VideoWorker(QThread)  ──frame, boxes, ts──┐
                                              ├─→ 主线程: 时间对齐 → 匈牙利匹配 → 渲染
    RadarWorker(QThread)  ──pkt────────────────┘

已内置的四件事：
    1. 时间戳对齐：雷达包的 t 是"雷达自己的时钟"，先估算 wall_offset，再按 |t_v - t_r| < 阈值 取最近邻
    2. 空间投影：project_radar_to_pixel() —— 现在是简化版（固定深度），有标定后只替换这一个函数
    3. 数据关联：代价矩阵 = 像素距离，门限过滤 + scipy 匈牙利
    4. 可视化：检测框 / 雷达点迹 / 匹配连线 / 右上角统计

用法：
    # 终端 1：假雷达
    python 20_fake_radar_server.py
    # 终端 2：界面（无视频文件时用内置合成画面，纯链路测试）
    python 31_gui_fusion.py --model yolov8s.pt
    python 31_gui_fusion.py --source D:/data/uav.mp4
    python 31_gui_fusion.py --source 0            # 摄像头
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections import deque

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QTextEdit, QVBoxLayout, QWidget,
)

# scipy 的匈牙利算法；没装就退化成贪心（结果会差一点，但链路能跑）
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MATCH_GATE_PX = 150.0     # 超过这个像素距离的配对直接拒绝（门限）
TIME_GATE_S = 0.15        # 图像与雷达帧的最大时间差


# ---------------------------------------------------------------- 空间投影
def project_radar_to_pixel(targets: list[dict], img_w: int, img_h: int,
                           focal: float = 900.0, depth_m: float = 20.0) -> list[dict]:
    """雷达坐标系(x 前, y 左, z 上, 单位 m) -> 像素坐标。

    这里用的是"固定深度"简化投影：假设所有目标都在 depth_m 米处，等效于把所有点压到一个平面上。
    它只用来让 demo 能看，物理上不严谨。真标定后换成：
        [u,v,1]^T = K · [R|t] · [X,Y,Z,1]^T          # 需要相机内参 K + 雷达→相机外参 R,t
    接口保持不变，只换这个函数内部，其余代码一行不用改。
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    out = []
    for tg in targets:
        u = cx + focal * tg["x"] / depth_m
        v = cy - focal * tg["y"] / depth_m
        if 0 <= u < img_w and 0 <= v < img_h:
            out.append({"id": tg.get("id", -1), "u": u, "v": v})
    return out


# ---------------------------------------------------------------- 视频 / 推理线程
class VideoWorker(QThread):
    frame_ready = pyqtSignal(object, list, float)   # frame(BGR), boxes, wall_ts

    def __init__(self, model_path: str, source: str, conf: float = 0.25,
                 infer_every: int = 1) -> None:
        super().__init__()
        self.model_path, self.source, self.conf = model_path, source, conf
        self.infer_every = max(1, infer_every)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _open(self):
        if self.source == "synthetic":
            return None, None
        if self.source.isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"打不开视频源: {self.source}")
        return cap, None

    def run(self) -> None:
        model = None
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
        except Exception as e:  # noqa: BLE001
            print(f"[video] 没有可用模型({e})，只显示画面不画框")

        try:
            cap, _ = self._open()
        except RuntimeError as e:
            print(e)
            cap = None

        idx = 0
        while not self._stop:
            if cap is None:
                # 合成画面：一个靶标绕圈，方便纯链路测试
                frame = np.full((720, 1280, 3), 30, np.uint8)
                server_t = time.time()
                x = 640 + 220 * np.cos(server_t * 0.8)
                y = 360 + 160 * np.sin(server_t * 0.8)
                cv2.circle(frame, (int(x), int(y)), 40, (0, 200, 255), -1)
                boxes: list = []
                if model is not None:
                    r = model(frame, conf=self.conf, verbose=False)[0]
                    boxes = [(*(int(v) for v in b), int(c), float(cf)) for b, cf, c in
                             zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
                self.frame_ready.emit(frame, boxes, time.time())
                self.msleep(33)
                continue

            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 循环播放，demo 时不至于停下
                continue
            ts = time.time()
            boxes = []
            if model is not None and idx % self.infer_every == 0:
                r = model(frame, conf=self.conf, verbose=False)[0]
                boxes = [(*(int(v) for v in b), int(c), float(cf)) for b, cf, c in
                         zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
                self._last_boxes = boxes
            elif model is not None:
                boxes = getattr(self, "_last_boxes", [])
            idx += 1
            self.frame_ready.emit(frame, boxes, ts)

        if cap is not None:
            cap.release()


# ---------------------------------------------------------------- 雷达线程
class RadarWorker(QThread):
    pkt_ready = pyqtSignal(dict)     # 已补上 wall_ts 字段

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.host, self.port = host, port
        self._stop = False
        self._offset: float | None = None    # wall_time - radar_time
        self.connected = pyqtSignal(bool)

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        buf = b""
        while not self._stop:
            try:
                s = socket.create_connection((self.host, self.port), timeout=3.0)
                s.settimeout(1.0)
                print(f"[radar] connected {self.host}:{self.port}")
                while not self._stop:
                    try:
                        chunk = s.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        raise ConnectionResetError("对端关闭")
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        pkt = json.loads(line.decode("utf-8"))
                        # 时钟对齐：第一帧估一次偏移，之后所有包换算到同一时间轴
                        if self._offset is None:
                            self._offset = time.time() - float(pkt.get("t", 0.0))
                        pkt["wall_ts"] = float(pkt.get("t", 0.0)) + self._offset
                        self.pkt_ready.emit(pkt)
            except (OSError, ConnectionResetError, json.JSONDecodeError) as e:
                print(f"[radar] 断开/异常: {e}，2s 后重连")
                time.sleep(2.0)


# ---------------------------------------------------------------- 主窗口
class FusionWindow(QMainWindow):
    def __init__(self, model_path: str, source: str, host: str, port: int) -> None:
        super().__init__()
        self.setWindowTitle("UAV 多源融合 Demo · 视觉 + 雷达")
        self.resize(1280, 780)

        self.view = QLabel("waiting for frames ...")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background:#111; color:#888;")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedWidth(330)

        body = QHBoxLayout()
        body.addWidget(self.view, 1)
        body.addWidget(self.log)
        central = QWidget()
        central.setLayout(body)
        self.setCentralWidget(central)

        self.radar_buf: deque[dict] = deque(maxlen=60)   # 最近 60 帧雷达
        self.last_img: np.ndarray | None = None
        self.names: dict = {}
        self.fps_t = time.time()
        self.fps_n = 0
        self.fps = 0.0

        self.video = VideoWorker(model_path, source)
        self.video.frame_ready.connect(self.on_frame)
        self.radar = RadarWorker(host, port)
        self.radar.pkt_ready.connect(self.on_radar)

        self.video.start()
        self.radar.start()
        self.log.append(f"video source = {source}")
        self.log.append(f"radar = {host}:{port}")
        if not HAS_SCIPY:
            self.log.append("[warn] 没装 scipy，退化为贪心匹配")

    # ---------------- 雷达回调
    def on_radar(self, pkt: dict) -> None:
        self.radar_buf.append(pkt)

    # ---------------- 取与图像时间最接近的雷达帧
    def pick_radar(self, video_ts: float) -> tuple[dict | None, float]:
        best, best_dt = None, 1e9
        for p in self.radar_buf:
            dt = abs(p["wall_ts"] - video_ts)
            if dt < best_dt:
                best, best_dt = p, dt
        if best is not None and best_dt <= TIME_GATE_S:
            return best, best_dt
        return None, best_dt

    # ---------------- 数据关联
    @staticmethod
    def associate(boxes: list, pts: list) -> list[tuple[int, int]]:
        """boxes: (x1,y1,x2,y2,cid,conf)  pts: {'id','u','v'}
        返回 [(box_idx, pt_idx), ...]"""
        if not boxes or not pts:
            return []
        C = np.zeros((len(boxes), len(pts)), dtype=float)
        for i, b in enumerate(boxes):
            bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            for j, p in enumerate(pts):
                C[i, j] = np.hypot(bx - p["u"], by - p["v"])

        pairs: list[tuple[int, int]] = []
        if HAS_SCIPY:
            rows, cols = linear_sum_assignment(C)
            for i, j in zip(rows, cols):
                if C[i, j] <= MATCH_GATE_PX:
                    pairs.append((int(i), int(j)))
        else:
            while True:
                i, j = np.unravel_index(np.argmin(C), C.shape)
                if C[i, j] > MATCH_GATE_PX:
                    break
                pairs.append((int(i), int(j)))
                C[i, :] = 1e9
                C[:, j] = 1e9
        return pairs

    # ---------------- 每帧渲染
    def on_frame(self, frame: np.ndarray, boxes: list, ts: float) -> None:
        if not self.names:
            try:
                from ultralytics import YOLO
                self.names = YOLO(self.video.model_path).names
            except Exception:  # noqa: BLE001
                pass

        h, w = frame.shape[:2]
        pkt, dt = self.pick_radar(ts)
        pts = project_radar_to_pixel(pkt["targets"], w, h) if pkt else []
        pairs = self.associate(boxes, pts)

        img = frame.copy()

        # 1) 视觉框
        for i, (x1, y1, x2, y2, cid, conf) in enumerate(boxes):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{self.names.get(cid, cid)} {conf:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 0), 1, cv2.LINE_AA)

        # 2) 雷达点迹（青色十字 + ID）
        for j, p in enumerate(pts):
            u, v = int(p["u"]), int(p["v"])
            cv2.drawMarker(img, (u, v), (255, 200, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, f"R{p['id']}", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 200, 0), 1, cv2.LINE_AA)

        # 3) 匹配连线（红：融合成功）
        for bi, pj in pairs:
            b, p = boxes[bi], pts[pj]
            cv2.line(img, (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)),
                     (int(p["u"]), int(p["v"])), (0, 0, 255), 2)

        # 4) 统计面板
        self.fps_n += 1
        if time.time() - self.fps_t >= 1.0:
            self.fps = self.fps_n / (time.time() - self.fps_t)
            self.fps_n, self.fps_t = 0, time.time()
        panel = [f"FPS {self.fps:5.1f}",
                 f"boxes {len(boxes):3d}",
                 f"radar {len(pts):3d}",
                 f"matched {len(pairs):3d}",
                 f"dt {dt * 1000:5.0f} ms"]
        for k, line in enumerate(panel):
            cv2.putText(img, line, (20, 30 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

        self.last_img = img
        self.show_img(img)

    def show_img(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, e) -> None:  # noqa: N802
        self.video.stop()
        self.radar.stop()
        self.video.wait(2000)
        self.radar.wait(2000)
        super().closeEvent(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--source", default="synthetic", help="视频路径 / 摄像头编号 / synthetic")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = FusionWindow(args.model, args.source, args.host, args.port)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
