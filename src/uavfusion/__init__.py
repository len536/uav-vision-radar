"""uavfusion —— 低空无人机检测与多源融合工具包。

模块地图（每个模块都可以 python -m uavfusion.<模块> 直接运行）:

环境与数据
    env_check         环境自检（torch/CUDA/ultralytics/PyQt5 ...）
    visdrone_convert  VisDrone 原始标注 -> YOLO 格式
    visdrone_verify   把标注框画回原图，肉眼核对标注质量

卡尔曼滤波（纯 NumPy，无滤波库依赖）
    kalman1d          Kalman1D 核心类（CV 运动模型，含逐步物理注释）
    kalman_experiments 1D 三组实验：静态/含噪跟踪/Q-R 调参对照
    kalman_track_box  跟踪检测框中心，含漏检段航迹维持实验

训练与评价
    train_visdrone    YOLOv8 训练（VisDrone，支持冒烟/正式两档参数）
    resume_train      断点续训（基于 last.pt）
    eval_predict      逐类 AP 评价 + 验证集预测可视化

雷达链路
    fake_radar_server 模拟毫米波雷达 TCP 服务端（20Hz，可注入丢帧）
    radar_client      TCP 客户端（'\n' 分帧、超时重连、帧率统计）

上位机界面（PyQt5）
    gui_image         静态图检测 GUI
    gui_fusion        视频 + 雷达融合 GUI（时间戳对齐/投影/匈牙利匹配）
"""

__version__ = "0.1.0"

from uavfusion.kalman1d import Kalman1D
from uavfusion.radar_client import RadarClient

__all__ = ["Kalman1D", "RadarClient", "__version__"]
