# -*- coding: utf-8 -*-
"""
皮蛋 · 语音助手（方案A：VOSK 唤醒 + 在线 DashScope 指令转写）
==============================================================
只在你说出唤醒词「皮蛋」之后，才把后面（或同一句里）的指令转成文字输出；
普通说话（没喊「皮蛋」）一律忽略，不输出任何文字。

    python pidan_assistant_v2.py

工作机制：
  - 麦克风只采一次音，同一份音频同时喂给两个引擎：
      1) VOSK（离线、免费）常开，只负责听唤醒词「皮蛋」（含同音字兜底）；
      2) 阿里云百炼 DashScope（在线、准）平时待机，唤醒后立刻启动，
         把「预录缓冲 + 后续语音」一起交给它，转写出准确的指令。
  - 输出一句指令后回到待机，等待下一次唤醒。

前置（一次性）：
  1) pip install dashscope
  2) 申请 DashScope API Key（见 speech_input_online.py 底部说明）
  3) （可选，方案B）python setup_hotword.py 创建「皮蛋」热词表

依赖：vosk + sounddevice + dashscope
"""

import collections
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

# ---------- 控制台编码与颜色 ----------
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
try:
    sys.stdin.reconfigure(encoding="utf-8")
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

# ---------- 配置区 ----------
SAMPLE_RATE = 16000
BLOCK_SIZE = 3200                # 每块约 0.2 秒
ONLINE_MODEL = "paraformer-realtime-v2"

BASE = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_DIR = os.path.join(BASE, "vosk-model-small-cn-0.22")
KEY_FILE = os.path.join(BASE, "dashscope_api_key.txt")
VOCAB_FILE = os.path.join(BASE, "dashscope_vocabulary_id.txt")
LOG_FILE = os.path.join(BASE, "transcript.txt")

# 唤醒词 + 常见同音字兜底（离线 VOSK 用它来判断“有没有喊皮蛋”）
WAKE_WORDS = ["皮蛋", "同学", "贾维斯", "星期五", "皮带",
              "皮担", "皮旦", "劈蛋", "批单", "披单", "皮单", "屁蛋"]

PREROLL_SECONDS = 2.0            # 唤醒后回放最近的音频秒数，防止“皮蛋+指令”一口气说丢开头
LISTEN_TIMEOUT = 5.0            # 唤醒后最多等多少秒，没指令就回待机
# ----------------------------

audio_q = queue.Queue()


# ---------- API Key / 热词 ----------
def load_api_key():
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        dashscope.api_key = key
        return
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            saved = f.read().strip()
        if saved:
            dashscope.api_key = saved
            return
    print(cyan("未检测到 DashScope API Key。"))
    try:
        key = input("请粘贴 API Key（sk-...）：").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not key:
        print(red("未输入 Key，程序退出。"))
        sys.exit(1)
    dashscope.api_key = key
    try:
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            f.write(key)
    except OSError:
        pass


def load_vocab_id():
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            v = f.read().strip()
        if v:
            return v
    return None


# ---------- 云端识别回调 ----------
def _is_sentence_end(sentence):
    if not isinstance(sentence, dict):
        return True
    try:
        if RecognitionResult.is_sentence_end(sentence):
            return True
    except Exception:
        pass
    return bool(sentence.get("sentence_end"))


class ASRCallback(RecognitionCallback):
    def __init__(self, result_q):
        self.result_q = result_q

    def on_open(self):
        self.result_q.put(("open", ""))

    def on_complete(self):
        self.result_q.put(("done", ""))

    def on_error(self, result):
        try:
            msg = result.message or str(result)
        except Exception:
            msg = str(result)
        self.result_q.put(("error", msg))

    def on_event(self, result):
        try:
            sentence = result.get_sentence()
        except Exception:
            return
        if not isinstance(sentence, dict):
            return
        text = sentence.get("text", "").replace(" ", "")
        if not text:
            return
        if _is_sentence_end(sentence):
            self.result_q.put(("final", text))
        else:
            self.result_q.put(("partial", text))


def remove_wake_words(text):
    """把句子里出现的唤醒词都去掉，剩下才是真正的指令。"""
    t = text
    for w in WAKE_WORDS:
        t = t.replace(w, "")
    return t.strip()


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
    vocab_id = load_vocab_id()

    print("正在加载本地唤醒模型...")
    vosk_model = Model(VOSK_MODEL_DIR)
    vosk_rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    print(bold(green("就绪。喊「皮蛋」唤醒我，然后说你的指令。Ctrl+C 退出。")))
    if vocab_id:
        print(dim("已加载热词表 vocabulary_id：" + vocab_id))
    else:
        print(dim("（未检测到热词表，可先运行 setup_hotword.py 创建，提升识别率）"))
    print()

    result_q = queue.Queue()
    sentences = []
    state = "idle"                 # idle=待机；listening=已唤醒等指令
    state_since = time.monotonic()
    recognition = None             # 云端识别对象，唤醒后才创建
    last_len = 0

    preroll = collections.deque(maxlen=int(PREROLL_SECONDS * SAMPLE_RATE / BLOCK_SIZE))

    def clear_partial():
        nonlocal last_len
        if last_len:
            sys.stdout.write("\r" + " " * last_len + "\r")
            sys.stdout.flush()
            last_len = 0

    def show_partial(text):
        nonlocal last_len
        line = f"{dim('识别中')}… {text}"
        pad = " " * max(0, last_len - len(line))
        sys.stdout.write("\r" + line + pad)
        sys.stdout.flush()
        last_len = len(line)

    def emit_command(text):
        clear_partial()
        cmd = remove_wake_words(text)
        if not cmd:
            return False
        sentences.append(cmd)
        print(f"{green('指令')} > {bold(cmd)}", flush=True)
        if LOG_FILE:
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] {cmd}\n")
            except OSError:
                pass
        return True

    def stop_dashscope():
        nonlocal recognition
        if recognition is not None:
            try:
                recognition.stop()
            except Exception:
                pass
            recognition = None

    def start_dashscope():
        nonlocal recognition
        cb = ASRCallback(result_q)
        kwargs = dict(model=ONLINE_MODEL, format="pcm",
                      sample_rate=SAMPLE_RATE, callback=cb)
        if vocab_id:
            kwargs["vocabulary_id"] = vocab_id
        r = Recognition(**kwargs)
        r.start()
        # 回放预录缓冲，让云端也能听到「皮蛋」后面的指令开头，不丢字
        for chunk in list(preroll):
            r.send_audio_frame(chunk)
        recognition = r

    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=BLOCK_SIZE,
                               device=None,
                               dtype="int16",
                               channels=1,
                               callback=audio_callback):
            while True:
                # ---- 1) 处理云端识别结果 ----
                while not result_q.empty():
                    kind, payload = result_q.get()
                    if kind == "partial":
                        if state == "listening":
                            show_partial(payload)
                    elif kind == "final":
                        if state == "listening":
                            if emit_command(payload):
                                stop_dashscope()
                                state = "idle"
                    elif kind == "error":
                        clear_partial()
                        print(red("识别出错：") + payload, file=sys.stderr)
                        stop_dashscope()
                        state = "idle"

                # ---- 2) 取音频 ----
                try:
                    data = audio_q.get(timeout=0.05)
                except queue.Empty:
                    if state == "listening" and \
                            time.monotonic() - state_since > LISTEN_TIMEOUT:
                        clear_partial()
                        print(dim("（超时未听到指令，回到待机）"))
                        stop_dashscope()
                        state = "idle"
                    continue

                # ---- 3) 喂离线 VOSK，检测唤醒词 ----
                if vosk_rec.AcceptWaveform(data):
                    vtext = json.loads(vosk_rec.Result()).get("text", "").replace(" ", "")
                else:
                    vtext = json.loads(vosk_rec.PartialResult()).get("partial", "").replace(" ", "")

                if state == "idle" and any(w in vtext for w in WAKE_WORDS):
                    state = "listening"
                    state_since = time.monotonic()
                    print(green("已唤醒，请说指令…"))
                    try:
                        start_dashscope()
                    except Exception as e:
                        print(red("启动在线识别失败："), e, file=sys.stderr)
                        state = "idle"

                # ---- 4) 预录缓冲 + 喂云端 ----
                preroll.append(data)
                if recognition is not None:
                    try:
                        recognition.send_audio_frame(data)
                    except Exception as e:
                        print(red("发送音频失败："), e, file=sys.stderr)
                        stop_dashscope()
                        state = "idle"

    except KeyboardInterrupt:
        clear_partial()
        print("\n已退出。本次共识别到 %d 条指令：" % len(sentences))
        for i, s in enumerate(sentences, 1):
            print(f"  {i}. {s}")
    except sd.PortAudioError as e:
        print(red("无法打开麦克风："), e, file=sys.stderr)
        print("提示：确认麦克风已接入，且在系统「声音设置」里是默认输入设备。",
              file=sys.stderr)
    finally:
        stop_dashscope()


if __name__ == "__main__":
    main()
