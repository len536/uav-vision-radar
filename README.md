# UAV 多源融合 · 第一阶段实操手册

> 版本：v1.1　｜　2026-11-03　｜　配套代码包：本目录（全部在本机实测通过）
> 定位：你桌面那份《学习计划 v1.0》的**落地执行版**——只讲"现在打开终端该敲什么"。

---

## 项目导航：我名下的三个仓库各管什么

| 仓库 | 一句话定位 | 里面放什么 |
|---|---|---|
| **uav-vision-radar**（本仓库） | 无人机检测+融合的**实验工具链** | 数据处理、训练/续训脚本、卡尔曼、假雷达链路、PyQt 可视化 |
| [drone-detector-service](https://github.com/len536/drone-detector-service) | 把训练好的模型**变成可调用的服务** | FastAPI 推理接口、Docker 镜像、PyQt 客户端 |
| [kalman-from-scratch](https://github.com/len536/kalman-from-scratch) | 纯 NumPy **手写卡尔曼滤波**+可视化 | 滤波器核心类、一维/跟踪实验、参数调优对照 |

分工逻辑：本仓库负责"从数据到模型"的完整实验；service 仓库负责"从模型到产品"的工程交付；kalman 仓库负责把融合核心算法单独讲清楚。

---

## 0. 先读：基于你机器实测的 4 条修正

我已经在你的电脑上实测过，原计划有 4 处必须改：

| # | 原计划 | 实测结果 | 修正 |
|:---:|---|---|---|
| 1 | 按无 N 卡/有 N 卡分方案 | **你有 RTX 5060 Laptop（8GB 显存，Blackwell sm_120）** | 本机训练完全可行，不需要 Colab/AutoDL |
| 2 | `pip install torch --index-url .../cu121` | **50 系显卡跑 cu121 会报 `no kernel image is available`** | 必须用 **cu128**（torch ≥ 2.7） |
| 3 | 用 conda 建环境 | 你机器**没装 conda**，有 Python 3.13 + uv | 直接用 venv（下面给了命令），Python 3.13 实测可用 |
| 4 | VisDrone 从 GitHub 官方下载 | 官方 Google Drive 国内经常连不上 | 首选 Kaggle 镜像（见 Step 3），备选 OpenDataLab |

**8GB 显存训练参数速查**（用 `nvidia-smi` 实时盯）：

| 模型 | imgsz | batch | 显存占用(约) | 用途 |
|---|:---:|:---:|:---:|---|
| yolov8n | 640 | 16 | ~3 GB | 冒烟/调流程 |
| yolov8s | 640 | 8 | ~5 GB | **主力配置** |
| yolov8s | 1280 | 2–4 | ~7 GB+ | 小目标涨点实验 |
| yolov8m | 640 | 4–6 | ~7 GB | 涨点实验 |

> 速度估算：yolov8s@640 batch8 约 1–1.5 min/epoch，50 epoch ≈ 1–1.5 小时 → **放夜里跑**。
> 白天写代码的时间不被训练占掉，这是你最重要的时间复用策略。

---

## 1. 优先级与 8 周时间线

### 1.1 优先级（和原计划一致，只调整执行方式）

1. **P0 视觉（YOLOv8+VisDrone）**——硬指标，占 ~45%
2. **P0 卡尔曼滤波**——理论核心，每天 1h 穿插，占 ~25%
3. **P1 PyQt+socket**——硬指标，**提前到 W2 开始搭骨架**（原计划 W5 太晚），占 ~20%
4. **P2 融合**——前三块的拼装，最后做，占 ~10%

**为什么 GUI 提前**：①"带界面的程序"是硬指标，拖到最后会和训练、融合抢时间；②GUI 骨架（30 号脚本）不依赖训练结果，W2 训练跑着的时候正好写 GUI。

### 1.2 时间线

| 周 | 主线（白天 2–2.5h） | 并行（1h） | 周末交付物 |
|:---:|---|---|---|
| W1 | 环境自检 → COCO 冒烟 → VisDrone 下载/转换/核对 → 官方权重零样本推理 | KF Day1–3（kalman_core + 3 个 demo） | 带框图片 + `fig/demo2_tracking.png` 三线图 |
| W2 | 3 epoch 冒烟训练 → 夜里正式训练 yolov8s@640×50ep → 学看 results.csv | **PyQt 最小窗口**（30_gui_image.py） | 自己的 VisDrone 权重 + mAP 截图 + 能开图画框的 GUI |
| W3 | 对照实验：640 vs 1280、n vs s、conf 阈值 → 填一张评估表 | `yolo track` 视频跟踪（拿 track id） | 对照实验表 + 帧级跟踪 demo |
| W4 | QThread 视频流 + FPS 优化；坐标转换（针孔模型手推 + `cv2.solvePnP` 验证） | socket TCP + 粘包处理（21 号脚本读懂） | 实时检测 GUI（视频/摄像头） |
| W5 | 假雷达接入：20→21 跑通，把雷达点画进 GUI | KF 接检测框中心（11 号脚本换成真 YOLO 输出） | "视觉框 + 雷达点迹"同屏 GUI |
| W6 | 时间戳对齐（最近邻 → 插值） | 门限过滤 + 匈牙利匹配 | 对齐 + 匹配脚本，匹配正确率抽查 |
| W7 | 三线程融合 demo 定稿（31 号脚本） | 决策级融合：关联成功提置信度 | **进组 demo v1** |
| W8 | 打磨 UI、写 README、准备 5 分钟组会汇报；缓冲 | — | demo + 汇报 PPT + 一页技术文档 |

### 1.3 每周末验收命令

```bash
# W1：能看到画框图片
yolo predict model=yolov8s.pt source=datasets/VisDrone/images/val imgsz=1280 conf=0.25
# W2 末：结果目录里有 results.csv / results.png / weights/best.pt
# W5 末：两个终端分别跑 20 和 30/31，界面同屏出现绿框和青色十字
```

---

## 2. 配套代码包（本目录，全部实测）

| 文件 | 作用 | 依赖 | 状态 |
|---|---|---|---|
| `00_check_env.py` | 环境自检，每次配完环境先跑它 | — | ✅ |
| `01_visdrone2yolo.py` | VisDrone→YOLO 标注转换（含 ignore 剔除/越界裁剪） | pillow | ✅ 编译通过 |
| `02_verify_labels.py` | **把标签画回原图肉眼核对**（防翻车关键步） | opencv | ✅ |
| `kalman_core.py` | 手写一维 KF 核心类（含马氏距离/丢帧处理） | numpy | ✅ |
| `10_kalman1d.py` | KF 三个可视化实验（K 收敛/三线图/Q-R 对照） | numpy+matplotlib | ✅ RMSE 0.56→0.26 |
| `11_kalman_track_box.py` | 检测框中心→KF 跟踪（视觉↔融合的接缝） | numpy+matplotlib | ✅ RMSE 8.5→5.4 px |
| `20_fake_radar_server.py` | 假雷达发送端（TCP+JSON 行分帧） | 标准库 | ✅ 20 Hz 实测 |
| `21_radar_client.py` | 接收端（分帧/重连/心跳统计） | 标准库 | ✅ 3s 收 60 帧 |
| `30_gui_image.py` | PyQt5 最小上位机：开图→YOLO→画框 | PyQt5+opencv+ultralytics | 编译通过 |
| `31_gui_fusion.py` | 三线程融合 demo 骨架（W6–W7 目标） | 同上+scipy | 编译通过 |

> 工作区里已建好 `.testvenv` 虚拟环境（D 盘）。装完 torch 后它就是完整环境，也可以按 Step 1 自己重建一个。

---

## 3. 方向一：YOLOv8 + VisDrone 保姆级步骤

### Step 0 确认硬件（已替你做完）

```
RTX 5060 Laptop GPU / 8151 MiB / sm_120 / 驱动 596.13 (CUDA 13.2)
```

### Step 1 建环境（10 分钟）

```powershell
# 在 D 盘建项目目录 + venv（uv 已装好，没有 uv 就用 python -m venv）
cd D:\
uv venv --python 3.13 D:\uav_lab\.venv
D:\uav_lab\.venv\Scripts\activate

# 关键：50 系必须 cu128（wheel 自带 CUDA 运行时，不用单独装 CUDA Toolkit）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python matplotlib scipy PyQt5 pillow

# 验证：必须打印 True 和 RTX 5060
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

装完跑 `python 00_check_env.py`，全绿再继续。

### Step 2 一分钟冒烟（不通不往下走）

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

成功 = 生成 `runs/detect/predict/bus.jpg` 且画了框。

### Step 3 下载 VisDrone2019-DET（约 2.5GB）

- **方案 A（推荐）**：Kaggle 数据集 `banuprasadb/visdrone-dataset`——**已转好 YOLO 格式并附带 visdrone.yaml**，可省掉 Step 4–5。用 kagglehub 拉取或网页下载。
- **方案 B**：OpenDataLab（国内直连）搜 VisDrone2019-DET。
- **方案 C**：官方 GitHub <https://github.com/VisDrone/VisDrone-Dataset>（需要梯子）。

> 即使用方案 A，也**强烈建议用方案 C 的原始标注跑一遍 Step 5 的转换脚本**——以后实验室自有数据集的标注转换是同样的套路，这一遍是练手。

### Step 4 目录结构

```
datasets/VisDrone/
├── images/{train,val,test}/        # *.jpg
└── labels/{train,val,test}/        # 转换后的 *.txt
```

### Step 5 标注转换 + 核对（最容易翻车的一步）

```bash
python 01_visdrone2yolo.py --root datasets/VisDrone --splits train,val
python 02_verify_labels.py --root datasets/VisDrone --split train --n 6
# 打开 out_verify/ 逐张看：框贴住目标 = 通过；整体偏移 = 归一化错了，回脚本里改
```

两个原计划没讲清的坑（脚本里已处理）：
1. **DET 静态图集的标注没有 frame_id**，第一个字段就是 bbox_left（VID 视频集才有 frame_id）。
2. **category=0 是 ignore region（大片背景）**，必须剔除，否则 mAP 莫名其妙地低。

### Step 6 数据配置 `visdrone.yaml`

```yaml
path: D:/uav_lab/datasets/VisDrone   # 绝对路径最稳
train: images/train
val: images/val
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

### Step 7 训练

```bash
# ① 冒烟（验证全流程，3 epoch 几分钟）
yolo detect train data=visdrone.yaml model=yolov8n.pt epochs=3 imgsz=320 batch=16 workers=2 project=runs_vd name=smoke exist_ok=True

# ② 正式训练（夜里跑）
yolo detect train data=visdrone.yaml model=yolov8s.pt epochs=50 imgsz=640 batch=8 workers=2 project=runs_vd name=s_640

# ③ 涨点实验（W3）
yolo detect train data=visdrone.yaml model=yolov8s.pt epochs=50 imgsz=1280 batch=4 workers=2 project=runs_vd name=s_1280

# ④ 零样本基线（训练前先跑，好知道"不训练能到多少"）
yolo predict model=yolov8s.pt source=datasets/VisDrone/images/val imgsz=1280 conf=0.25
```

**合理预期**（量级参考）：yolov8s@640 在 VisDrone val 上 mAP50 大约 0.35–0.42、mAP50-95 大约 0.18–0.22；换 1280 通常还能 +3~5 mAP50。VisDrone 小目标极多，别指望 0.6+，论文里 mAP50-95 0.2+ 就是正常水平。

### Step 8 看懂输出（融合的起点）

| 产物 | 看什么 |
|---|---|
| `results.csv` | `metrics/mAP50(B)`、`metrics/mAP50-95(B)` 是否还在涨；`val/box_loss` 很早回升 = 过拟合 |
| `results.png` | 损失下降 + mAP 上升的大形态 |
| `confusion_matrix.png` | 错分集中在哪两类（通常是 car/van、people/pedestrian） |
| `save_txt=True` 的推理输出 | 每行 `cls conf x1 y1 x2 y2`（归一化） |

**Python 读取检测结果（务必吃透，这就是融合的数据流起点）**：

```python
from ultralytics import YOLO
m = YOLO("runs_vd/s_640/weights/best.pt")
r = m("某张图.jpg", conf=0.25)[0]
for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
    print(m.names[int(cls)], round(conf, 3), [round(v, 1) for v in box])
```

### Step 9 常见报错速查

| 报错/现象 | 原因与解决 |
|---|---|
| `no kernel image is available` | 装成 cu121 了 → 重装 cu128 |
| `CUDA out of memory` | batch 减半 / imgsz 降；1280 时 batch≤4 |
| `No labels found` | labels 路径错或 txt 与 jpg 不同名 |
| DataLoader worker 报错（Windows 常见） | `workers=2` 或 `workers=0` |
| 框位置偏移 | 标注没归一化，或写成了 `x1 y1 w h`（应为 `cx cy w h`） |
| mAP 恒为 0 | yaml 的 names 顺序与 class_id 不一致；或标签全是空文件 |

---

## 4. 方向二：一维卡尔曼滤波（3 天 + 已验证代码）

### 4.1 路线

- **Day 1（2h）**：直觉 + 公式。读 [kalmanfilter.net](https://www.kalmanfilter.net/default.aspx) 一维两章（每章有数值算例）+ B 站 DR_CAN《卡尔曼滤波器》系列前 4 讲。验收：合上书默写 5 个式子（见 `kalman_core.py` 文件头注释）。
- **Day 2（2h）**：手写。**不要抄框架**，对着 5 个式子自己写 `kalman_core.py` 那样的类，然后跑 `python 10_kalman1d.py`，和你手写的对照。
- **Day 3（2h）**：做实验。跑 `python 11_kalman_track_box.py`；然后做两个问题：① Q 放大 10 倍曲线怎么变？② 把目标机动改猛（正弦幅度 2m、频率 0.8Hz）为什么滤波反而比原始观测差？（答案：CV 模型假设失效——这正是 CA 模型/IMM 存在的理由，组会讲这个很加分）

### 4.2 已经替你验证过的结论（直接可用）

| 实验 | 结果 | 说明 |
|---|---|---|
| 三线图 | RMSE 0.56 → 0.26（改善 53%） | `10_kalman1d.py` demo2 |
| 框中心跟踪（含 10 帧漏检） | RMSE 8.5 → 5.4 px（改善 37%） | `11_kalman_track_box.py` |
| Q/R 扫描 | Q 大→跟得快但抖；R 大→平滑但滞后 | `10_kalman1d.py` demo3 |

**调参心法**（很多人卡在这）：
- `R` = 观测噪声方差，可直接估计：对静止目标连续测 N 次，算方差。
- `q` ≈ 目标加速度标准差的平方，量纲看着"离谱"是正常的；实际用网格搜索（1e3→1e7）画 RMSE 曲线取谷底。
- 速度的初始协方差要给大（如 300²），否则滤波器不敢改速度，会一直滞后。

### 4.3 教程推荐（按上手顺序）

| 顺序 | 资源 | 为什么 |
|:---:|---|---|
| 1 | [kalmanfilter.net](https://www.kalmanfilter.net/default.aspx) | 一维讲得最透，每章带数值例子，跟手写代码一一对应 |
| 2 | B 站 **DR_CAN《卡尔曼滤波器》系列** | 中文最经典，从直觉→推导→代码，适合通勤看 |
| 3 | GitHub **rlabbe/Kalman-and-Bayesian-Filters-in-Python** | 免费电子书 + 可运行 Jupyter，第 1 章 g-h 滤波是 KF 的"去矩阵版"，理解极深 |
| 4 | 博客 **Bzarg: How a Kalman filter works, in pictures** | 图解多维矩阵版，为 4x4 做铺垫 |
| 5 | `filterpy` 源码（`kalman/kalman_filter.py`） | 对照自己的 30 行，看工程版多了什么 |
| 6 | 书：《卡尔曼滤波原理及应用》黄小平（薄，中文）；*Probabilistic Robotics* 第 3 章（进阶） | 组里大概率有纸版 |

> 坐标转换（像素↔相机↔世界）放 W4：手推针孔模型 → 用 `cv2.solvePnP` 对照验证。

---

## 5. 方向三：PyQt + socket 知识清单

### 5.1 PyQt 必会（每条都有"验收动作"）

| # | 知识点 | 验收动作 |
|:---:|---|---|
| 1 | QApplication / QMainWindow / QWidget 三层关系 | 能说清为什么 `app.exec_()` 是死循环 |
| 2 | **信号与槽** | 写一个按钮点一下往日志里加一行 |
| 3 | **QThread**（核心） | 推理放进 worker 线程，点"检测"时窗口拖动不卡 |
| 4 | 图像显示 | ndarray→QImage：显式给 `bytesPerLine=3*w` 且 `.copy()` |
| 5 | 自定义信号 | worker 里 `pyqtSignal(dict)` 把结果抛回主线程 |
| 6 | 控件 | QLabel / QPushButton / QSlider / QTextEdit / QFileDialog |
| 7 | 布局 | QHBoxLayout/QVBoxLayout 嵌套出"左图右日志" |
| 8 | 生命周期 | 重写 `closeEvent`：`thread.quit(); thread.wait()` |

**三大坑**（30/31 号脚本里都有注释）：① BGR/RGB 颜色错乱；② 推理放主线程卡死；③ QImage 不拷贝内存导致偶发花屏。

### 5.2 socket 必会

| # | 知识点 | 验收动作 |
|:---:|---|---|
| 1 | TCP vs UDP | 先 TCP；雷达实际协议问清再补 UDP/组播 |
| 2 | API 全流程 | `socket/connect/bind/listen/accept/sendall/recv/close` |
| 3 | **TCP 粘包**（最高频 bug） | `recv(4096)` 可能半条消息 → '\n' 分帧或"4 字节长度头+包体"；`21_radar_client.py` 的 buf 写法背下来 |
| 4 | `struct` 模块 | `struct.unpack('<5f', data[:20])` 解二进制帧（真雷达多半是这个） |
| 5 | 超时与重连 | `settimeout` + try/except + 自动重连（21 号已实现） |
| 6 | 集成进 Qt | QThread 里跑 recv 循环，信号把 dict 抛主线程 |
| 7 | 统计时 | 帧率 Hz、dt 均值/抖动（21 号每秒打印的那行） |

**无硬件自测链路**（已实测 20 Hz 稳定）：

```bash
# 终端 1
python 20_fake_radar_server.py --hz 20 --targets 3
# 终端 2
python 21_radar_client.py          # 每秒打印帧率/目标数/坐标
# 终端 3（装好 ultralytics 后）
python 31_gui_fusion.py             # 界面上：绿框=视觉，青十字=雷达，红线=匹配成功
```

### 5.3 必须问雷达同事的 5 个问题

1. 协议：TCP / UDP / 组播？端口？
2. 数据格式：JSON / ASCII / 二进制？字段顺序、字节序？
3. 单位：米还是厘米？速度是相对雷达还是绝对？
4. 坐标系定义：x 前 y 左 z 上？还是 NED？
5. 时间戳：雷达本地时钟还是主机授时？精度多少 ms？（直接决定 W6 时间对齐方案）

---

## 6. 方向四：多源融合（W6–W8）

- **时间对齐**：先做最近邻（|t_v − t_r| < 0.15s 收一个）；进阶做线性插值到统一时间轴。`31_gui_fusion.py` 里 `pick_radar()` 就是最近邻版。
- **空间对齐**：无标定数据前，用 `project_radar_to_pixel()` 的简化投影（固定深度）让链路先跑通；拿到内外参后**只替换这一个函数**。
- **数据关联**：代价矩阵 = 框中心与雷达点的像素距离 → 门限（150px）过滤 → `scipy.optimize.linear_sum_assignment`。贪心版也写在里面做对照。
- **决策级融合**：关联成功 → 置信度提高/标记"双源确认"；单源目标 → 标"可疑"。
- **杀手锏（加分项）**：VisDrone 有视频序列，`yolo track model=best.pt source=xx.mp4` 直接给 track id → 每条轨迹挂一个 `Kalman1D` → 平滑轨迹 + 漏检外推 → 再和雷达关联。这就是从"检测 demo"到"跟踪融合系统"的一步。

---

## 7. 纪律（原计划 6 条全部保留，追加 2 条）

7. **训练一律放夜里跑**，白天时间写代码/分析结果，绝不盯着进度条。
8. **每个实验先跑基线再改**：零样本 → 默认参数 → 单变量改动，每次只动一个参数，否则结果没法解释。

---

## 8. 资料汇总

- Ultralytics 文档：<https://docs.ultralytics.com/modes/train/>（Train/Predict/Track 三页精读）
- VisDrone：<https://github.com/VisDrone/VisDrone-Dataset>；Kaggle 镜像 `banuprasadb/visdrone-dataset`
- 参考 baseline 仓库：`Mohd-Abdul-Rafay/small-object-detection-visdrone`（YOLOv8l VisDrone 完整实验记录，可对照 mAP）
- KF：kalmanfilter.net ／ DR_CAN ／ rlabbe/Kalman-and-Bayesian-Filters-in-Python ／ Bzarg 图解
- Qt for Python：<https://doc.qt.io/qtforpython-6/>（PyQt5 教程同样适用，API 几乎一致）

---

## 9. 今天就开始的三件事

1. Step 1 建环境 → `python 00_check_env.py` 全绿。
2. Step 2 冒烟测试出带框图片。
3. KF Day 1：读 kalmanfilter.net 一维两章，合书默写 5 个式子。
