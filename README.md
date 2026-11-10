# uav-vision-radar

低空无人机检测与多源融合（毫米波雷达 + 视觉）工具包：VisDrone 训练链路、纯 NumPy 卡尔曼滤波、模拟雷达 TCP 链路、PyQt5 上位机。

> 本仓库是三仓库矩阵中的**算法实验场**（数据 → 模型 → 雷达 → 界面的完整过程）：

| 仓库 | 分工 |
|---|---|
| **uav-vision-radar**（本仓库） | 算法实验场：数据转换、训练/续训/评价、卡尔曼、假雷达链路、PyQt 界面 |
| [drone-detector-service](https://github.com/len536/drone-detector-service) | 工程交付：YOLOv8 → FastAPI 推理服务 → Docker 化 |
| [kalman-from-scratch](https://github.com/len536/kalman-from-scratch) | 算法讲解：纯 NumPy 手写卡尔曼 + 可视化实验 |

## 功能模块

| 模块 | 说明 |
|---|---|
| 环境自检 | 校验 torch/CUDA/算力/ultralytics/PyQt5，RTX 50 系（Blackwell sm_120）专项提示 |
| 数据处理 | VisDrone 原始标注 → YOLO 格式转换；标注框画回原图肉眼核对 |
| 卡尔曼滤波 | 纯 NumPy 手写 1D CV 模型（2 状态），含 Q/R 调参、漏检段航迹维持实验 |
| 训练链路 | YOLOv8 训练 / 断点续训 / 逐类 AP 评价 + 预测可视化 |
| 雷达链路 | 模拟 20Hz 毫米波雷达 TCP 服务端（可注入丢帧）+ 抗粘包客户端 |
| 上位机 | PyQt5 静态图检测 GUI；视频 + 雷达点融合 GUI（时间戳对齐 / 坐标投影 / 匈牙利匹配） |

## 快速开始

```bash
git clone git@github.com:len536/uav-vision-radar.git
cd uav-vision-radar

python -m venv .venv && .venv\Scripts\activate

# torch 安装（二选一）：
#   RTX 50 系必须用 cu128 及以上（sm_120，装 cu121 会报 no kernel image）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   其他显卡 / CPU：pip install torch torchvision

pip install -e .
uav-check-env        # 环境自检，全绿即可开始
```

安装后所有功能以命令行入口提供：

| 命令 | 用途 |
|---|---|
| `uav-check-env` | 环境自检 |
| `uav-visdrone-convert` | VisDrone 标注转 YOLO 格式 |
| `uav-visdrone-verify` | 标注框画回原图核对 |
| `uav-kalman-demo` | 1D 卡尔曼三组实验（静态 / 跟踪 / Q-R 对照） |
| `uav-kalman-track` | 检测框中心跟踪（含 10 帧漏检航迹维持） |
| `uav-train` | YOLOv8 训练 |
| `uav-resume` | 断点续训 |
| `uav-eval` | 逐类 AP 评价 + 预测可视化 |
| `uav-radar-server` | 模拟雷达 TCP 服务端（20Hz） |
| `uav-radar-client` | 雷达客户端（'\n' 分帧 / 超时重连 / 帧率统计） |
| `uav-gui-image` | 静态图检测 GUI |
| `uav-gui-fusion` | 视频 + 雷达融合 GUI |

也可以 `python -m uavfusion.<模块名>` 运行。数据集与权重不入库（见 `.gitignore`），训练配置在 `configs/`。

## 目录结构

```
uav-vision-radar/
├── src/uavfusion/          # 全部代码（可安装的包）
│   ├── env_check.py        # 环境自检
│   ├── visdrone_convert.py # 标注格式转换
│   ├── visdrone_verify.py  # 标注可视化核对
│   ├── kalman1d.py         # Kalman1D 核心类（逐步物理注释）
│   ├── kalman_experiments.py / kalman_track_box.py
│   ├── train_visdrone.py / resume_train.py / eval_predict.py
│   ├── fake_radar_server.py / radar_client.py
│   └── gui_image.py / gui_fusion.py
├── configs/                # 训练数据集 yaml
├── fig/                    # 实验输出图
└── pyproject.toml          # 12 个命令行入口定义
```

## 实测结果（RTX 5060 Laptop 8GB / torch 2.9.1+cu128）

**卡尔曼滤波（纯 NumPy）**

| 实验 | 原始观测 RMSE | 滤波后 | 改善 |
|---|---|---|---|
| 1D 含噪轨迹跟踪 | 0.556 | 0.261 | **-53.2%** |
| 检测框中心跟踪（含连续 10 帧漏检，凭状态预测维持航迹） | 8.5 px | 5.4 px | **-37%** |

**VisDrone YOLOv8 训练**

| 配置 | mAP50 | mAP50-95 | 备注 |
|---|---|---|---|
| yolov8s, 320px, 3 ep（冒烟） | 0.110 | 0.053 | 只验证链路 |
| yolov8s, 640px, 26 ep | **0.366** | **0.209** | VisDrone-val，小目标密集场景 |

分辨率 320→640 使 mAP50 提升约 3 倍：VisDrone 目标普遍仅十几像素，`imgsz` 是该数据集第一优先调参项。

**雷达链路**：模拟 20Hz 服务端 + 客户端实测稳定收发（'\n' 分帧、3 秒 60 帧无粘包）。

## 已知坑位（踩过，写在这里）

1. **RTX 50 系（sm_120）必须装 cu128 轮子**，cu121 直接 `no kernel image`。
2. **VisDrone DET 标注没有 frame_id 字段**，第一个字段就是 bbox_left——网上大量教程抄错。
3. **`category=0`（忽略区域）/ 10（其他）必须剔除**，否则训练目标被污染。
4. **VisDrone 单图最多 900+ 目标**，超过 YOLO 默认 `max_det=300`，验证召回会被硬截断。
5. **VisDrone 检测的是地面目标（相机在无人机上）**，没有 drone 类别；检测无人机本体需换数据集（如 Anti-UAV / 自采数据）。

## Roadmap

- [x] 环境自检 + 数据转换核对 + YOLOv8 训练/续训/评价
- [x] 1D 卡尔曼 + 漏检航迹维持
- [x] 模拟雷达 20Hz TCP 链路 + 抗粘包客户端
- [x] PyQt 静态检测 GUI + 融合 GUI（时间戳对齐 / 投影 / 匈牙利匹配骨架）
- [ ] 640 vs 1280 分辨率对照实验、`conf` 阈值扫描
- [ ] 帧级跟踪（`yolo track`）+ 卡尔曼航迹平滑接入 GUI
- [ ] 真实雷达协议适配（TCP/UDP、二进制解析、坐标系定义）
- [ ] 时间戳对齐 + 数据关联进融合 GUI 实测 → 进组 demo

## License

[MIT](LICENSE)
