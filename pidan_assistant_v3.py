# -*- coding: utf-8 -*-
"""
皮蛋 · 语音助手 v3（VOSK 唤醒 + 在线 DashScope 指令转写）
==========================================================
只在喊「皮蛋」之后才转写指令，平时说话一律忽略。

    python pidan_assistant_v3.py

唤醒后的流程：
  1. 保持至少 2 秒聆听，等你开口说第一句；
  2. 听到第一句后，进入「转化」状态，开始攒指令；
  3. 说话停顿（间隔）只要不超过 1 秒，就继续攒；
  4. 连续 1 秒没再说话 → 结束本次对话，把攒下的指令一次性输出。

依赖：vosk + sounddevice + dashscope
"""

import json
import os
import queue
import sys
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer

try:
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
except ImportError:
    print("缺少依赖 dashscope，请先执行：pip install dashscope")
    sys.exit(1)

# ---------- 控制台 ----------
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
def cyan(t):   return _c(t, "96")

# ---------- 配置 ----------
SAMPLE_RATE = 16000
BLOCK_SIZE = 3200            # 每块 0.2 秒
ONLINE_MODEL = "paraformer-realtime-v2"

BASE = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_DIR = os.path.join(BASE, "vosk-model-small-cn-0.22")
KEY_FILE = os.path.join(BASE, "dashscope_api_key.txt")
LOG_FILE = os.path.join(BASE, "transcript.txt")

WAKE_WORDS = ["皮蛋", "同学", "贾维斯", "星期五", "皮带",
              "皮担", "皮旦", "劈蛋", "批单", "披单", "皮单", "屁蛋"]
PUNCT = "，。！？、,.!?;；:：…"

LISTEN_TIMEOUT = 2.0         # 唤醒后至少聆听 2 秒等第一句
SILENCE_GAP = 1.0           # 说话间隔最多允许 1 秒
# ----------------------------

audio_q = queue.Queue()


# ---------- API Key ----------
def load_api_key():
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE, encoding="utf-8") as f:
            key = f.read().strip()
    if not key:
        print(cyan("未检测到 DashScope API Key，请粘贴（sk-...）："))
        key = input().strip()
    dashscope.api_key = key
    try:
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            f.write(key)
    except OSError:
        pass


# ---------- 文本处理 ----------
def is_content(text):
    """去掉空白、唤醒词、标点后还有字 → 是真实内容（不是空句/句号）。"""
    t = text.replace(" ", "")
    for w in WAKE_WORDS:
        t = t.replace(w, "")
    for p in PUNCT:
        t = t.replace(p, "")
    return t.strip() != ""


def clean_text(text):
    """去掉空白和唤醒词，返回真正的指令文本。"""
    t = text.replace(" ", "")
    for w in WAKE_WORDS:
        t = t.replace(w, "")
    return t.strip()


# ---------- 云端回调 ----------
def _ended(sentence):
    try:
        return RecognitionResult.is_sentence_end(sentence)
    except Exception:
        return bool(sentence.get("sentence_end"))


class ASRCallback(RecognitionCallback):
    def __init__(self, q):
        self.q = q

    def on_open(self):
        pass

    def on_complete(self):
        pass

    def on_error(self, result):
        self.q.put(("error", getattr(result, "message", str(result))))

    def on_event(self, result):
        sentence = result.get_sentence() or {}
        text = sentence.get("text", "").replace(" ", "")
        if text:
            self.q.put(("final" if _ended(sentence) else "partial", text))


# ---------- 音频采集回调 ------------- 
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(bytes(indata))


# ---------- 主程序 ----------
def main():
    if not os.path.isdir(VOSK_MODEL_DIR):
        print(red(f"找不到 VOSK 模型：{VOSK_MODEL_DIR}"), file=sys.stderr)
        sys.exit(1)

    load_api_key()
    print("加载本地唤醒模型...")
    vosk_model = Model(VOSK_MODEL_DIR)
    vosk_rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    print(bold(green(f"就绪。喊「皮蛋」唤醒我：聆听 {LISTEN_TIMEOUT:.0f} 秒，停顿 {SILENCE_GAP:.0f} 秒结束。")))
    print(dim("Ctrl+C 退出。"))
    print()

    result_q = queue.Queue()
    history = []               # 已完成的指令（退出时汇总用）
    collected = []             # 本次对话攒下的句子
    state = "idle"             # idle / listening / transcribing
    last_activity = 0.0        # 最近一次收到语音内容的时刻
    recognition = None


    def show_listening():
        line = f"{dim('识别中')}…"
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def clear_partial():
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def start_cloud():
        nonlocal recognition
        recognition = Recognition(model=ONLINE_MODEL, format="pcm",
                                  sample_rate=SAMPLE_RATE,
                                  callback=ASRCallback(result_q))
        recognition.start()
        show_listening()

    def stop_cloud():
        nonlocal recognition
        if recognition is not None:
            recognition.stop()
            recognition = None

    def finish():
        nonlocal state
        clear_partial()
        stop_cloud()
        if collected:
            command = "".join(collected)
            history.append(command)
            print(f"{green('指令')} > {bold(command)}", flush=True)
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] {command}\n")
            except OSError:
                pass
            collected.clear()
        state = "idle"

    def cancel():
        nonlocal state
        clear_partial()
        stop_cloud()
        collected.clear()
        state = "idle"
        print(dim("（未听到指令，回到待机）"))

    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                               device=None, dtype="int16", channels=1,
                               callback=audio_callback):
            while True:
                now = time.monotonic()

                # 1) 处理云端结果
                while not result_q.empty():
                    kind, payload = result_q.get()
                    if kind == "partial":
                        if state != "idle" and is_content(payload):
                            last_activity = now
                    elif kind == "final":
                        if state != "idle" and is_content(payload):
                            last_activity = now
                            collected.append(clean_text(payload))
                            state = "transcribing"
                    elif kind == "error":
                        print(red("识别出错：") + payload, file=sys.stderr)
                        cancel()

                # 2) 超时判断
                if state == "listening" and now - last_activity > LISTEN_TIMEOUT:
                    cancel()
                elif state == "transcribing" and now - last_activity > SILENCE_GAP:
                    finish()

                # 3) 取音频
                try:
                    data = audio_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                # 4) 喂 VOSK，检测唤醒词
                if vosk_rec.AcceptWaveform(data):
                    vtext = json.loads(vosk_rec.Result()).get("text", "").replace(" ", "")
                else:
                    vtext = json.loads(vosk_rec.PartialResult()).get("partial", "").replace(" ", "")

                if state == "idle" and any(w in vtext for w in WAKE_WORDS):
                    state = "listening"
                    last_activity = time.monotonic()
                    print(green("已唤醒，请说指令…"))
                    try:
                        start_cloud()
                    except Exception as e:
                        print(red("启动在线识别失败："), e, file=sys.stderr)
                        state = "idle"

                # 5) 喂云端
                if recognition is not None:
                    try:
                        recognition.send_audio_frame(data)
                    except Exception:
                        print(red("发送音频失败，回到待机。"), file=sys.stderr)
                        cancel()

    except KeyboardInterrupt:
        clear_partial()
        print(f"\n已退出。本次共 {len(history)} 条指令：")
        for i, s in enumerate(history, 1):
            print(f"  {i}. {s}")
    except sd.PortAudioError as e:
        print(red("无法打开麦克风："), e, file=sys.stderr)
        print("提示：确认麦克风已接入，且是系统默认输入设备。", file=sys.stderr)
    finally:
        stop_cloud()


if __name__ == "__main__":
    main()
