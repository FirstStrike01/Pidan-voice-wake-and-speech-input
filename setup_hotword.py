# -*- coding: utf-8 -*-
"""
创建阿里云百炼「定制热词」词表（方案B）
======================================
把「皮蛋」等唤醒词加进在线识别引擎，让云端更倾向把这些音认对。
运行一次即可，之后 pidan_assistant_v2.py 会自动读取并带上热词表。

    python setup_hotword.py

会做两件事：
  1) 调用 API 创建词表；
  2) 把 vocabulary_id 存到 dashscope_vocabulary_id.txt。

前置：pip install dashscope，且已配置 API Key（可先跑一次 speech_input_online.py）。
"""

import os
import sys

import dashscope
from dashscope import Vocabulary

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
if os.name == "nt":
    os.system("")

BASE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(BASE, "dashscope_api_key.txt")
VOCAB_FILE = os.path.join(BASE, "dashscope_vocabulary_id.txt")

TARGET_MODEL = "paraformer-realtime-v2"
PREFIX = "pidan"

# 唤醒词 + 同音字，weight 越大越「偏袒」，范围一般 1~10
WORDS = [
    {"text": "皮蛋", "weight": 5, "lang": "zh"},
    {"text": "皮带", "weight": 5, "lang": "zh"},
    {"text": "皮旦", "weight": 5, "lang": "zh"},
    {"text": "皮担", "weight": 5, "lang": "zh"},
    {"text": "皮单", "weight": 5, "lang": "zh"},
    {"text": "批单", "weight": 5, "lang": "zh"},
    {"text": "披单", "weight": 5, "lang": "zh"},
    {"text": "屁蛋", "weight": 5, "lang": "zh"},
    {"text": "劈蛋", "weight": 5, "lang": "zh"},
]


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
    print("未检测到 DashScope API Key。")
    print("请先运行 speech_input_online.py 配置一次，或直接粘贴：")
    try:
        key = input("API Key（sk-...）：").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not key:
        sys.exit(1)
    dashscope.api_key = key
    try:
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            f.write(key)
    except OSError:
        pass


def main():
    load_api_key()
    print(f"正在创建热词表（target_model={TARGET_MODEL}）...")
    try:
        result = Vocabulary.create(
            prefix=PREFIX,
            target_model=TARGET_MODEL,
            words=WORDS,
        )
    except Exception as e:
        print("创建失败：", e)
        print("提示：确认 API Key 有效、账号已开通语音识别；若仍失败，把报错贴给皮蛋。")
        sys.exit(1)

    out = getattr(result, "output", None)
    vocab_id = None
    if isinstance(out, dict):
        vocab_id = out.get("vocabulary_id")
    elif out is not None:
        vocab_id = getattr(out, "vocabulary_id", None)

    if not vocab_id:
        print("创建已提交，但没从返回里读到 vocabulary_id，原始返回：")
        print(result)
        sys.exit(1)

    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        f.write(vocab_id)
    print("热词表创建成功 ✅")
    print("vocabulary_id =", vocab_id)
    print("已保存到", VOCAB_FILE)


if __name__ == "__main__":
    main()
