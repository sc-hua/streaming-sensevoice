# streaming-sensevoice

Streaming SenseVoice processes inference in chunks of [SenseVoice](https://github.com/FunAudioLLM/SenseVoice).

## Usage

```bash
pip install -r requirements.txt

# run unified HTTP + WS server (default port 9000)
python server.py --host 0.0.0.0 --port 9000
```

## 接口文档

本项目提供 HTTP 和 WebSocket 两种接口。

详细接口定义请参考 [API.md](API.md)。

- **HTTP 接口**: 支持短音频文件的同步转写。
- **WebSocket 接口**: 支持流式音频的实时转写，包含 VAD 事件推送。

## 配置参数

可以通过命令行参数或环境变量配置服务：

- `SENSEVOICE_MODEL_PATH`: 模型路径
- `FSMN_VAD_MODEL_PATH`: VAD 模型路径
- `DEVICE`: 推理设备 (cpu/cuda)
- `SAMPLERATE`: 采样率 (默认 16000)
- `CHUNK_MS`: 分块时长 (默认 200ms)

更多配置项请查看 `API.md` 或运行 `python server.py --help`。

