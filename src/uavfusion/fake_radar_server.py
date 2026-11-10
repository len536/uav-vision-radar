"""
20_fake_radar_server.py —— 假雷达发送端（没硬件时的一切前提）

协议（自己定的，拿到真雷达文档后按实际协议改这一个文件即可）：
    TCP，每条消息 = UTF-8 JSON + '\n'（换行分帧，最省事的粘包解法）
    {
      "t": 12.34,                     # 秒，发送端时间戳
      "targets": [
        {"id": 1, "x": 3.2, "y": -1.1, "vx": 0.5, "vy": 0.0, "rcs": 0.8}
      ]
    }
    单位统一用米、米/秒；坐标系：x 前、y 左、z 上（右手系）—— 跟实验室确认后改这里。

用法：
    python 20_fake_radar_server.py                 # 默认 127.0.0.1:9000，20Hz
    python 20_fake_radar_server.py --port 9001 --hz 10 --targets 3
    python 20_fake_radar_server.py --drop 0.1      # 10% 的帧丢掉，测试客户端鲁棒性

配套：另开一个终端跑 python 21_radar_client.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time


def make_targets(n: int, t: float) -> list[dict]:
    out = []
    for i in range(n):
        # 每条目标一个匀速/圆周运动，带噪声，模拟真实点迹抖动
        if i % 2 == 0:
            x = 2.0 + 0.6 * t
            y = -3.0 + 0.3 * i
            vx, vy = 0.6, 0.0
        else:
            r = 5.0 + i
            w = 0.4
            x = r * math.cos(w * t)
            y = r * math.sin(w * t)
            vx, vy = -r * w * math.sin(w * t), r * w * math.cos(w * t)
        out.append({
            "id": i + 1,
            "x": round(x + random.gauss(0, 0.05), 3),
            "y": round(y + random.gauss(0, 0.05), 3),
            "vx": round(vx, 3),
            "vy": round(vy, 3),
            "rcs": round(random.uniform(0.3, 1.5), 2),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--hz", type=float, default=20.0, help="发送频率")
    ap.add_argument("--targets", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.0, help="随机丢帧概率 0~1")
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[server] 监听 {args.host}:{args.port} ，等待客户端连接... (Ctrl+C 退出)")

    conn, addr = srv.accept()
    print(f"[server] 客户端已连接 {addr}，开始 {args.hz} Hz 推送 {args.targets} 个目标")

    period = 1.0 / args.hz
    t0 = time.time()
    sent = 0
    next_tick = time.time()
    try:
        while True:
            t = time.time() - t0
            if random.random() >= args.drop:
                pkt = {"t": round(t, 4), "targets": make_targets(args.targets, t)}
                conn.sendall((json.dumps(pkt) + "\n").encode("utf-8"))
                sent += 1
            next_tick += period
            sleep = next_tick - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.time()  # 落后了就重置节奏，避免雪崩
    except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        print(f"\n[server] 断开，共发送 {sent} 帧")
    finally:
        try:
            conn.close()
        finally:
            srv.close()


if __name__ == "__main__":
    main()
