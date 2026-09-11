# -*- coding: utf-8 -*-
"""
语音输入（联网 · 阿里云百炼 DashScope 实时识别）
================================================
和 speech_input.py 用法一样：运行后对着麦克风说话，实时转成文字，
说完一句（停顿一下）就定稿输出一句，并缓存写入 transcript.txt。

    python speech_input_online.py

区别：识别走阿里云百炼 DashScope 的 paraformer-realtime-v2（联网、更准、
流式实时），而不是本地 VOSK。断网 / 没申请 Key 时，可用 speech_input.py 兜底。

前置（一次性）：
  1) pip install dashscope
  2) 申请 DashScope API Key（见文件底部说明），首次运行会提示粘贴并自动保存。

依赖：dashscope + sounddevice（sounddevice 已装）。
"""

import os
import queue
import sys
import time

import sounddevice as sd

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
    os.system("")  # Windows 控制台启用 ANSI 颜色

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t):  return _c(t, "92")
def dim(t):    return _c(t, "2")
def bold(t):   return _c(t, "1")
def red(t):    return _c(t, "91")
def cyan(t):   return _c(t, "96")

# ---------- 配置区 ----------
MODEL = "paraformer-realtime-v2"   # 百炼实时语音识别模型
SAMPLE_RATE = 16000                # 采样率（固定 16k）
BLOCK_SIZE = 3200                  # 每次发送约 0.2 秒音频，延迟更低
SHOW_PARTIAL = True                # 是否实时显示中间识别结果

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "transcript.txt")

KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dashscope_api_key.txt")

# 唤醒词同音字兜底：开头是这些就自动去掉，只保留指令内容（和 speech_input.py 一致）。
# 想保留原始结果就把它设成 []。
STRIP_WAKE_WORDS = ["皮蛋", "同学", "贾维斯", "星期五", "皮带",
                    "皮担", "皮旦", "劈蛋", "批单", "披单", "皮单", "屁蛋"]
# ----------------------------


# ---------- API Key ----------
def load_api_key():
    """依次从环境变量 DASHSCOPE_API_KEY、本地文件读取；都没有则提示粘贴并保存。"""
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
    print(cyan("获取方式：登录 https://bailian.console.aliyun.com/ 创建（见文件底部说明）。"))
    try:
        key = input("请粘贴你的 DashScope API Key（sk-...）：").strip()
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
        print(green(f"已保存 Key 到 {os.path.basename(KEY_FILE)}（仅存本机，下次免输入）。\n"))
    except OSError:
        pass


# ---------- 识别回调 ----------
def _is_sentence_end(sentence):
    """判断这一句是否已说完整（定稿）。"""
    if not isinstance(sentence, dict):
        return True
    try:
        if RecognitionResult.is_sentence_end(sentence):
            return True
    except Exception:
        pass
    return bool(sentence.get("sentence_end"))


class ASRCallback(RecognitionCallback):
    """DashScope 识别结果回调（在 SDK 自己的线程里触发，这里只往队列里丢）。"""

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


# ---------- 文本处理 ----------
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


# ---------- 主程序 ----------
def main():
    load_api_key()

    print("正在连接阿里云百炼实时语音识别...")
    result_q = queue.Queue()
    callback = ASRCallback(result_q)
    recognition = Recognition(
        model=MODEL,
        format="pcm",
        sample_rate=SAMPLE_RATE,
        callback=callback,
    )

    try:
        recognition.start()
    except Exception as e:
        print(red("启动识别失败（请检查 API Key / 网络 / 免费额度）："), e, file=sys.stderr)
        sys.exit(1)

    print(bold(green("就绪！请对着麦克风说话，说完一句稍作停顿即可输出。")))
    print(dim("按 Ctrl+C 退出。"))
    if LOG_FILE:
        print(dim(f"每句会同时写入：{LOG_FILE}"))
    print()

    audio_q = queue.Queue()
    sentences = []      # 本次会话缓存的所有定稿句子
    last_len = 0
    error_seen = [False]

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

    def on_sentence(text):
        clear_partial()
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

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_q.put(bytes(indata))

    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=BLOCK_SIZE,
                               device=None,
                               dtype="int16",
                               channels=1,
                               callback=audio_callback):
            while True:
                # 1) 处理识别结果（由 SDK 回调线程入队）
                while not result_q.empty():
                    kind, payload = result_q.get()
                    if kind == "open":
                        print(dim("已连接云端识别服务，开始监听…"))
                    elif kind == "partial":
                        if SHOW_PARTIAL:
                            show_partial(payload)
                    elif kind == "final":
                        on_sentence(payload)
                    elif kind == "error":
                        clear_partial()
                        print(red("识别出错：") + payload, file=sys.stderr)
                        error_seen[0] = True

                if error_seen[0]:
                    break

                # 2) 把麦克风音频发给云端识别
                try:
                    data = audio_q.get(timeout=0.05)
                    recognition.send_audio_frame(data)
                except queue.Empty:
                    pass
                except Exception as e:
                    print(red("发送音频失败："), e, file=sys.stderr)
                    break

    except KeyboardInterrupt:
        clear_partial()
        print("\n已退出。本次共识别到 %d 句：" % len(sentences))
        for i, s in enumerate(sentences, 1):
            print(f"  {i}. {s}")
    except sd.PortAudioError as e:
        print(red("无法打开麦克风："), e, file=sys.stderr)
        print("提示：确认麦克风已接入，且在系统「声音设置」里是默认输入设备。",
              file=sys.stderr)
    finally:
        try:
            recognition.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()


# =====================================================================
# 如何申请 DashScope API Key（阿里云百炼）：
#   1. 打开 https://bailian.console.aliyun.com/ ，用支付宝/淘宝登录阿里云账号；
#      首次需要完成个人实名认证。
#   2. 打开「API-KEY 管理」（直达链接：
#      https://bailian.console.aliyun.com/?apiKey=1 ）。
#   3. 点「创建 API-KEY」，复制生成的 sk-... 字符串。
#   4. 首次运行本脚本会提示粘贴，之后自动保存到 dashscope_api_key.txt。
#   语音识别 paraformer-realtime-v2 新用户有免费额度，超出按量计费（单价很低）。
# =====================================================================
