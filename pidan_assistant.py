# -*- coding: utf-8 -*-
"""
皮蛋 · 语音助手主循环（唤醒 + 听一句指令）
==========================================
把“关键词唤醒”和“语音输入”串成一个整体：

    python pidan_assistant.py

流程：
  1. 一直监听，等你喊唤醒词“皮蛋”（含同音字兜底）；
  2. 听到唤醒词后，屏幕提示“已唤醒”，进入“听指令”状态；
  3. 你说一句话（指令），说完停顿，程序就把这句话定稿输出；
  4. 输出后回到待机，等待下一次唤醒，循环往复。

和 speech_input.py 的区别：
  - speech_input.py 无差别地把每句话都转成文字（更适合测识别准确率）；
  - 本脚本只在“唤醒之后”才输出你说的话，适合直接做“贾维斯”式随聊入口。

依赖：vosk + sounddevice。模型：同目录 vosk-model-small-cn-0.22。
退出：Ctrl+C。
"""

import json
import os
import queue
import sys
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------- 控制台编码与颜色 ----------
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

if os.name == "nt":
    os.system("")

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t):  return _c(t, "92")
def dim(t):    return _c(t, "2")
def bold(t):   return _c(t, "1")
def red(t):    return _c(t, "91")

# ---------- 配置区 ----------
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "vosk-model-small-cn-0.22")

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "transcript.txt")

# 唤醒词 + 常见同音字兜底
WAKE_WORDS = ["皮蛋", "同学", "贾维斯", "星期五", "皮带",
              "皮担", "皮旦", "劈蛋", "批单", "披单", "皮单", "屁蛋"]

LISTEN_TIMEOUT = 8.0   # 唤醒后若超过这么多秒没听到指令，就自动回到待机
# ----------------------------

q = queue.Queue()


def callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def remove_wake_words(text):
    """把句子里出现的唤醒词都去掉，剩下的就是指令内容。"""
    t = text
    for w in WAKE_WORDS:
        t = t.replace(w, "")
    return t.strip()


def on_command(text):
    """唤醒后收到的一句指令：缓存输出 + 写日志。

    以后接 DeepSeek：把 text 发给大模型即可（见文件底部示例）。
    """
    text = remove_wake_words(text)
    if not text:
        return
    print(f"{green('指令')} > {bold(text)}", flush=True)
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
        except OSError as e:
            print(red(f"（写日志失败：{e}）"), file=sys.stderr)


def main():
    if not os.path.isdir(MODEL_DIR):
        print(red(f"找不到模型目录：{MODEL_DIR}"), file=sys.stderr)
        print("请先运行 setup_model.py 下载模型。", file=sys.stderr)
        sys.exit(1)

    print("正在加载语音模型...")
    model = Model(MODEL_DIR)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    print(bold(green("皮蛋已就绪。喊“皮蛋”唤醒我，然后说你的指令。Ctrl+C 退出。")))
    print(dim(f"唤醒后 {LISTEN_TIMEOUT:.0f} 秒内没听到指令会自动回到待机。"))
    print()

    state = "idle"            # idle=待机等唤醒；listening=已唤醒，等指令
    state_since = time.monotonic()
    last_len = 0

    def clear_partial():
        nonlocal last_len
        if last_len:
            sys.stdout.write("\r" + " " * last_len + "\r")
            sys.stdout.flush()
            last_len = 0

    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=BLOCK_SIZE,
                               device=None,
                               dtype="int16",
                               channels=1,
                               callback=callback):
            while True:
                try:
                    data = q.get(timeout=0.2)
                except queue.Empty:
                    if state == "listening" and \
                            time.monotonic() - state_since > LISTEN_TIMEOUT:
                        clear_partial()
                        print(dim("（超时未听到指令，回到待机）"))
                        state = "idle"
                    continue

                if rec.AcceptWaveform(data):
                    # 一句话结束（定稿结果）
                    text = json.loads(rec.Result()).get("text", "").replace(" ", "")
                    if not text:
                        continue

                    if state == "idle":
                        # 待机：只在听到唤醒词时才动作
                        if any(w in text for w in WAKE_WORDS):
                            cmd = remove_wake_words(text)
                            print(green("已唤醒"))
                            if cmd:
                                # “皮蛋 帮我查天气”这种一句话里直接带指令
                                on_command(cmd)
                            else:
                                # 只喊了“皮蛋”，进入听指令状态
                                state = "listening"
                                state_since = time.monotonic()
                                print(dim("请说指令…"))
                    else:
                        # 已唤醒：把这一句当成指令
                        cmd = remove_wake_words(text)
                        clear_partial()
                        if cmd:
                            on_command(cmd)
                        state = "idle"
                else:
                    # 实时中间结果：只在听指令时显示
                    partial = json.loads(rec.PartialResult()).get("partial", "").replace(" ", "")
                    if state == "listening" and partial:
                        line = f"{dim('识别中')}… {partial}"
                        pad = " " * max(0, last_len - len(line))
                        sys.stdout.write("\r" + line + pad)
                        sys.stdout.flush()
                        last_len = len(line)

    except KeyboardInterrupt:
        clear_partial()
        print("\n已退出。")
    except sd.PortAudioError as e:
        print(red("无法打开麦克风："), e, file=sys.stderr)
        print("提示：确认麦克风已接入，且在系统「声音设置」里是默认输入设备。",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


# =====================================================================
# 接入 DeepSeek（下一步）：把 on_command 里识别的 text 发给大模型即可。
# deepseek_chat.py 在上级目录，示例：
#
#   import sys as _sys, os as _os
#   _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
#   import deepseek_chat
#   deepseek_chat.load_api_key()
#   for _piece in deepseek_chat.stream_chat(
#           "deepseek-chat", [{"role": "user", "content": text}]):
#       _sys.stdout.write(_piece); _sys.stdout.flush()
#   _sys.stdout.write("\n")
# =====================================================================
