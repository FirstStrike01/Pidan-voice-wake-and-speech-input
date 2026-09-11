# -*- coding: utf-8 -*-
"""
语音输入（语音转文字 / STT）
=============================
运行后持续监听麦克风，把你说的话实时转成文字，说完一句就输出一句：

    python speech_input.py

效果：
  - 说话过程中，屏幕最下面会实时刷新中间识别结果（让你知道它在听）；
  - 你停顿、说完一句话后，程序立刻把这句话定稿，独占一行输出；
  - 每句话都会缓存到内存，并追加写入 transcript.txt（可关）；
  - 按 Ctrl+C 退出，退出时会打印本次会话识别到的所有句子。

依赖：vosk + sounddevice（都已安装）。
模型：同目录下的 vosk-model-small-cn-0.22（已下载好）。
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
    os.system("")  # Windows 控制台启用 ANSI 颜色

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t):  return _c(t, "92")
def dim(t):    return _c(t, "2")
def bold(t):   return _c(t, "1")
def red(t):    return _c(t, "91")

# ---------- 配置区 ----------
SAMPLE_RATE = 16000          # vosk 固定 16000
BLOCK_SIZE = 8000            # 每次喂给识别器的采样点数（约 0.5 秒）
SHOW_PARTIAL = True          # 是否实时显示“正在识别”的中间结果

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "vosk-model-small-cn-0.22")

# 每句定稿后追加写入这个文件（相当于持久化缓存）；不想写文件就改成 None
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "transcript.txt")

# 唤醒词同音字兜底：如果一句话的开头是“皮蛋/同学”等唤醒词，就自动去掉，
# 只保留真正的指令内容（例如“皮蛋 帮我查天气” -> “帮我查天气”）。
# 只想看原始识别结果时，把它设成 [] 即可。
STRIP_WAKE_WORDS = ["皮蛋", "同学", "贾维斯", "星期五", "皮带",
                    "皮担", "皮旦", "劈蛋", "批单", "披单", "皮单", "屁蛋"]
# ----------------------------

q = queue.Queue()
sentences = []     # 内存缓存：本次会话所有定稿后的句子


def callback(indata, frames, time_info, status):
    """麦克风每来一块音频，就丢进队列。"""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def strip_wake(text):
    """去掉开头的唤醒词，只保留指令内容。"""
    if not STRIP_WAKE_WORDS:
        return text
    t = text
    while t:
        hit = None
        for w in STRIP_WAKE_WORDS:
            if t.startswith(w) and len(t) > len(w):
                hit = w
                break
        if hit is None:
            break
        t = t[len(hit):].strip()
    return t


def on_sentence(text):
    """每定稿一句就调用一次：缓存 + 输出 + 写文件。

    以后要把这句发给 DeepSeek，就在这里把 text 传给
    deepseek_chat.stream_chat(...)（见文件底部说明）。
    """
    text = strip_wake(text)
    if not text:
        return
    sentences.append(text)
    print(f"{green('识别')} > {bold(text)}", flush=True)
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
    print(bold(green("就绪！请对着麦克风说话，说完一句稍作停顿即可输出。")))
    print(dim("按 Ctrl+C 退出。"))
    if LOG_FILE:
        print(dim(f"每句会同时写入：{LOG_FILE}"))
    print()

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
                data = q.get()

                if rec.AcceptWaveform(data):
                    # 一句话结束（定稿结果）
                    clear_partial()
                    text = json.loads(rec.Result()).get("text", "").replace(" ", "")
                    if text:
                        on_sentence(text)
                else:
                    # 实时中间结果
                    if SHOW_PARTIAL:
                        partial = json.loads(rec.PartialResult()).get("partial", "").replace(" ", "")
                        if partial:
                            line = f"{dim('识别中')}… {partial}"
                            pad = " " * max(0, last_len - len(line))
                            sys.stdout.write("\r" + line + pad)
                            sys.stdout.flush()
                            last_len = len(line)

    except KeyboardInterrupt:
        clear_partial()
        print("\n已退出。本次共识别到 %d 句：" % len(sentences))
        for i, s in enumerate(sentences, 1):
            print(f"  {i}. {s}")
    except sd.PortAudioError as e:
        print(red("无法打开麦克风："), e, file=sys.stderr)
        print("提示：确认麦克风已接入，且在系统「声音设置」里是默认输入设备。",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


# =====================================================================
# 接入 DeepSeek（下一步）：在 on_sentence 里把 text 发给大模型即可。
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
