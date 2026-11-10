"""
30_gui_image.py —— PyQt5 最小上位机（第 1 步：打开图片 → YOLO → 画框 → 显示）

刻意做小：只有一个窗口 + 一个后台线程。这个骨架跑通后，31_gui_live.py 才有意义。

必须理解的三件事（这就是全部难点）：
    1. 推理不能放在 UI 线程 —— 否则点"检测"后窗口白屏/转圈。
       -> 用 QThread 子类，run() 里干活，算完 emit 信号。
    2. 图像格式：OpenCV 是 BGR，Qt 默认 RGB。用 QImage.Format_BGR888 或先 cvtColor。
    3. QImage 不拷贝底层内存 —— numpy 数组一释放图就花屏。必须 .copy()。
       （网上 90% 的 PyQt+YOLO 代码在这里埋了雷，偶尔花屏就是这么来的）

用法：
    python 30_gui_image.py
    python 30_gui_image.py --model yolov8s.pt

依赖：PyQt5、opencv-python、ultralytics
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QTextEdit, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------- 工具函数
def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    """BGR ndarray -> QPixmap（关键：bytesPerLine 显式给 3*w，并 copy 一份）"""
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
    return QPixmap.fromImage(qimg)


def draw_boxes(img: np.ndarray, boxes, names: dict) -> np.ndarray:
    """boxes: [(x1,y1,x2,y2,cls_id,conf), ...]"""
    out = img.copy()
    for x1, y1, x2, y2, cid, conf in boxes:
        color = (0, 200, 0) if conf > 0.5 else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cid, cid)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- 推理线程
class InferWorker(QThread):
    """QThread 用法要点：把活写在 run() 里；结果用信号抛回主线程。"""

    done = pyqtSignal(object, object, str)   # (原图, boxes, 日志文本)
    failed = pyqtSignal(str)

    def __init__(self, model_path: str, img_path: str, conf: float) -> None:
        super().__init__()
        self.model_path, self.img_path, self.conf = model_path, img_path, conf

    def run(self) -> None:
        try:
            from ultralytics import YOLO

            t0 = time.perf_counter()
            model = YOLO(self.model_path)
            t_load = (time.perf_counter() - t0) * 1000

            img = cv2.imread(self.img_path)
            if img is None:
                self.failed.emit(f"读不到图片: {self.img_path}")
                return

            t1 = time.perf_counter()
            r = model(img, conf=self.conf, verbose=False)[0]
            t_infer = (time.perf_counter() - t1) * 1000

            boxes = []
            for box, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                x1, y1, x2, y2 = (int(v) for v in box)
                boxes.append((x1, y1, x2, y2, int(cl), float(cf)))

            msg = (f"加载模型 {t_load:.0f} ms | 推理 {t_infer:.0f} ms | "
                   f"检出 {len(boxes)} 个目标 | imgsz={r.orig_shape}")
            self.done.emit(img, boxes, msg)
        except Exception as e:  # noqa: BLE001  界面上要把错误显示出来，不能静默
            self.failed.emit(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 主窗口
class MainWindow(QMainWindow):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.setWindowTitle("UAV 检测上位机 · Step1 单图")
        self.resize(1150, 720)

        self.model_path = model_path
        self.img_path: str | None = None
        self.worker: InferWorker | None = None
        self.names: dict = {}
        self.last_img: np.ndarray | None = None

        # --- 控件 ---
        self.btn_open = QPushButton("打开图片")
        self.btn_run = QPushButton("检测")
        self.btn_run.setEnabled(False)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 95)
        self.slider.setValue(25)          # conf = 0.25
        self.lbl_conf = QLabel("conf 0.25")
        self.view = QLabel("请选择一张图片")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(800, 600)
        self.view.setStyleSheet("background:#222; color:#aaa;")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(110)

        top = QHBoxLayout()
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_run)
        top.addSpacing(12)
        top.addWidget(self.lbl_conf)
        top.addWidget(self.slider, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("运行日志"))
        right.addWidget(self.log)

        body = QHBoxLayout()
        body.addWidget(self.view, 3)
        body.addLayout(right, 1)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(body)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # --- 信号槽 ---
        self.btn_open.clicked.connect(self.on_open)
        self.btn_run.clicked.connect(self.on_run)
        self.slider.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"conf {v / 100:.2f}"))

    # 选图
    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        self.img_path = path
        self.btn_run.setEnabled(True)
        img = cv2.imread(path)
        if img is not None:
            self.last_img = img
            self.show_img(img)
        self.log.append(f"加载: {path}")

    # 推理（起线程，不要在 UI 线程里算）
    def on_run(self) -> None:
        if not self.img_path or (self.worker and self.worker.isRunning()):
            return
        self.btn_run.setEnabled(False)
        conf = self.slider.value() / 100
        self.worker = InferWorker(self.model_path, self.img_path, conf)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.start()

    def on_done(self, img, boxes, msg: str) -> None:
        from ultralytics import YOLO  # 只为拿类别名，读一次很便宜
        try:
            if not self.names:
                self.names = YOLO(self.model_path).names
        except Exception:  # noqa: BLE001
            pass
        self.last_img = img
        self.show_img(draw_boxes(img, boxes, self.names))
        self.log.append(msg)
        self.setWindowTitle(f"UAV 检测上位机 · Step1 单图  [{len(boxes)} 个目标]")

    def on_failed(self, msg: str) -> None:
        self.log.append(f"[错误] {msg}")

    def show_img(self, img: np.ndarray) -> None:
        pm = bgr_to_pixmap(img)
        self.view.setPixmap(pm.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # 窗口缩放时重绘，否则图会被拉伸变形
    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if self.last_img is not None:
            self.show_img(self.last_img)

    # 关窗时收线程，否则进程退不干净（PyQt 常见坑）
    def closeEvent(self, e) -> None:  # noqa: N802
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)
        super().closeEvent(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="本地权重或模型名")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    w = MainWindow(args.model)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
