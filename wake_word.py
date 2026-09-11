# -*- coding: utf-8 -*-
"""
语音唤醒词检测

运行后对着麦克风说出关键词（默认 "皮蛋"），识别到就 print(1)。

- 完全离线识别，不联网、不调任何云服务
- 依赖：vosk + sounddevice（麦克风采集）
- 退出：按 Ctrl + C

想换关键词？改下面 WAKE_WORDS 列表即可，比如改成 ["你好", "开始"]。
"""

import json
import os
import queue
import sys
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------- 配置区 ----------
# 唤醒词 + 常见同音字兜底。
# 原理：vosk 听写"皮蛋"这类生僻词时，容易输出别的同音字。
# 把同音字一起列进来，无论被听成哪个同音字，都算命中。
WAKE_WORDS = ["皮蛋", "同学",  "星期五", "皮带",
              "皮担", "皮旦", "劈蛋", "批单", "披单", "皮单", "屁蛋"]
SAMPLE_RATE = 16000            # 采样率（vosk 固定用 16000）
BLOCK_SIZE = 8000              # 每次喂给识别器的采样点数（约 0.5 秒）
COOLDOWN = 1.5                 # 命中后的冷却秒数，避免一句话刷出多个 1
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "vosk-model-small-cn-0.22")
# ----------------------------

q = queue.Queue()


def callback(indata, frames, time_info, status):
    """麦克风每来一块音频，就丢进队列。"""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"找不到模型目录：{MODEL_DIR}", file=sys.stderr)
        print("请先运行 setup_model.py 下载模型。", file=sys.stderr)
        sys.exit(1)

    print("正在加载语音模型...")
    model = Model(MODEL_DIR)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    print("就绪！请对着麦克风说话（默认唤醒词：%s），按 Ctrl+C 退出。"
          % "、".join(WAKE_WORDS))

    last_hit = 0.0

    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=BLOCK_SIZE,
                               device=None,
                               dtype="int16",
                               channels=1,
                               callback=callback):
            while True:
                data = q.get()

                # AcceptWaveform 返回 True 表示一句话结束（final 结果），
                # 否则用 partial（实时中间结果）判断，延迟更低。
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result()).get("text", "")
                else:
                    text = json.loads(rec.PartialResult()).get("partial", "")

                text = text.replace(" ", "")
                if any(w in text for w in WAKE_WORDS):
                    now = time.monotonic()
                    if now - last_hit >= COOLDOWN:
                        last_hit = now
                        print(1, flush=True)
                        rec.Reset()   # 清空识别状态，避免同一句话反复触发

    except KeyboardInterrupt:
        print("\n已退出。")
    except sd.PortAudioError as e:
        print("无法打开麦克风：", e, file=sys.stderr)
        print("提示：确认麦克风已接入，且在系统「声音设置」里是默认输入设备。",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
