"""
21_radar_client.py —— 雷达接收端（真雷达接进来时，只改 parse_frame 一个函数）

三件事在这里定型，以后接真硬件不用重写：
    1. 分帧：TCP 是字节流，recv(4096) 可能只收到半条消息 —— 用 '\n' 切分 + 保留残余 buffer。
    2. 重连：对端断开（实验室雷达重启/网线松）时自动重试，程序不能崩。
    3. 回落格式：拿到真雷达文档后，把 parse_frame 换成 struct.unpack 解析二进制帧即可。

用法：
    # 终端 1
    python 20_fake_radar_server.py
    # 终端 2
    python 21_radar_client.py
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import time


class RadarClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, on_frame=None) -> None:
        self.host, self.port = host, port
        self.on_frame = on_frame          # 回调：收到一帧就调用 on_frame(pkt)
        self.sock: socket.socket | None = None
        self.buf = b""
        self.running = False

    # ---------- 连接（含重连） ----------
    def connect(self, retry: int = 3) -> socket.socket:
        for i in range(retry):
            try:
                s = socket.create_connection((self.host, self.port), timeout=3.0)
                s.settimeout(None)
                self.sock = s
                print(f"[radar] connected {self.host}:{self.port}")
                return s
            except OSError as e:
                print(f"[radar] 连接失败({i + 1}/{retry}): {e}")
                time.sleep(1.0)
        raise ConnectionError("连不上雷达端口，确认 server 在跑、端口/防火墙没挡")

    # ---------- 解析：换协议只改这里 ----------
    @staticmethod
    def parse_frame(payload: bytes) -> dict:
        """文本协议：JSON。二进制协议示例：
            t, x, y, vx, vy = struct.unpack('<5f', payload[:20])   # 小端 float32
        具体字段顺序、字节序一律以雷达厂家文档为准。
        """
        return json.loads(payload.decode("utf-8"))

    # ---------- 主循环 ----------
    def run_forever(self) -> None:
        self.running = True
        while self.running:
            try:
                s = self.sock or self.connect()
                while self.running:
                    chunk = s.recv(4096)
                    if not chunk:
                        raise ConnectionResetError("对端关闭")
                    self.buf += chunk
                    while b"\n" in self.buf:
                        line, self.buf = self.buf.split(b"\n", 1)
                        if line.strip():
                            pkt = self.parse_frame(line)
                            if self.on_frame:
                                self.on_frame(pkt)
            except (ConnectionResetError, OSError) as e:
                print(f"[radar] 断开: {e}，2 秒后重连")
                self.sock = None
                time.sleep(2.0)

    def stop(self) -> None:
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    stamps: list[float] = []
    count = 0
    t_last = time.time()
    last_pkt: dict | None = None

    def on_frame(pkt: dict) -> None:
        nonlocal count, last_pkt
        count += 1
        stamps.append(time.time())
        last_pkt = pkt

    cli = RadarClient(args.host, args.port, on_frame=on_frame)
    print("[radar] Ctrl+C 退出")

    # 每秒打印一次统计，这是调试链路的"心跳"
    import threading

    th = threading.Thread(target=cli.run_forever, daemon=True)
    th.start()

    try:
        while True:
            time.sleep(1.0)
            now = time.time()
            dt = now - t_last
            hz = count / dt if dt > 0 else 0.0
            jitter = ""
            if len(stamps) >= 3:
                d = [b - a for a, b in zip(stamps[-20:], stamps[-19:])]
                if d:
                    jitter = f" dt均值 {statistics.mean(d) * 1000:.1f}ms 抖动±{statistics.pstdev(d) * 1000:.1f}ms"
            n_tg = len(last_pkt["targets"]) if last_pkt else 0
            print(f"[radar] {hz:5.1f} Hz | 累计 {count} 帧 | 目标 {n_tg} 个{jitter}")
            if last_pkt and last_pkt.get("targets"):
                t0 = last_pkt["targets"][0]
                print(f"        t={last_pkt['t']:.2f}s  id={t0['id']}  "
                      f"x={t0['x']:+.2f} y={t0['y']:+.2f} vx={t0['vx']:+.2f}")
            count = 0
            t_last = now
    except KeyboardInterrupt:
        cli.stop()
        print("\n[radar] bye")


if __name__ == "__main__":
    main()
