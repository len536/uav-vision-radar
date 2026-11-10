"""
00_check_env.py —— 环境自检（每次新建环境后第一件事就是跑它）

用法：
    python 00_check_env.py

只做检查，不修改任何东西。全绿才往下走。
"""
import importlib.util
import platform
import sys


def has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def main() -> None:
    print("=" * 62)
    print(f"Python  : {sys.version.split()[0]}  ({platform.architecture()[0]})")
    print(f"平台    : {platform.platform()}")
    print(f"解释器  : {sys.executable}")
    print("=" * 62)

    ok = True

    # ---------- 1. torch + CUDA ----------
    if not has(torch_name := "torch"):
        print("[X] torch 未安装")
        ok = False
    else:
        import torch  # noqa: PLC0415

        print(f"[{'OK' if torch.cuda.is_available() else 'X '}] torch {torch.__version__}")
        print(f"     CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"     设备    : {name}")
            print(f"     算力    : sm_{cap[0]}{cap[1]}")
            print(f"     显存    : {total:.1f} GB")
            if cap[0] == 12:
                print("     -> Blackwell(50 系)：必须用 cu128 及以上的 torch，cu121 会报 no kernel image")
        else:
            print("     -> 没有 CUDA：训练会用 CPU，VisDrone 训练请换 Kaggle/Colab 或租卡")
            ok = False

    # ---------- 2. 其他依赖 ----------
    for mod, pkg, why in [
        ("ultralytics", "ultralytics", "YOLOv8 训练/推理"),
        ("cv2", "opencv-python", "读图/画框/视频"),
        ("numpy", "numpy", "矩阵运算"),
        ("matplotlib", "matplotlib", "画 KF 曲线"),
        ("scipy", "scipy", "匈牙利算法 linear_sum_assignment"),
        ("PyQt5", "PyQt5", "上位机界面"),
        ("PIL", "pillow", "读图片尺寸"),
    ]:
        mark = "OK" if has(mod) else "X "
        print(f"[{mark}] {pkg:<16} {why}")
        if mark == "X " and mod in {"ultralytics", "numpy", "cv2"}:
            ok = False

    print("=" * 62)
    if ok:
        print("核心环境可用。下一步：见 README 第 3 节 Step 2 冒烟测试。")
    else:
        print("有缺失。执行：uv pip install ultralytics opencv-python matplotlib scipy PyQt5 pillow")
    print("=" * 62)


if __name__ == "__main__":
    main()
