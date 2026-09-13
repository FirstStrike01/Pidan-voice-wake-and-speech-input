# 皮蛋·第一步 语音助手 · 唤醒词 + 语音输入

喊「皮蛋」唤醒，然后说话，自动转成文字输出。

## 原理

- **唤醒词**：VOSK 离线模型（免费、本地运行、带同音字兜底）
- **语音转写**：阿里云百炼 DashScope（联网、实时、准确）

## 怎么跑

1. 安装依赖：`pip install vosk sounddevice dashscope`
2. 申请阿里云百炼 API Key（首次运行会提示粘贴并自动保存）
3. `python pidan_assistant_v3.py`

## 文件

- `wake_word.py` —— 离线唤醒词检测
- `speech_input.py` —— 离线语音输入（VOSK）
- `speech_input_online.py` —— 在线语音输入（DashScope）
- `pidan_assistant_v3.py` —— 主程序（唤醒 + 指令转写）
- `CHANGELOG.md` —— 更新日志
