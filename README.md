# streaming-sensevoice

Streaming SenseVoice processes inference in chunks of [SenseVoice](https://github.com/FunAudioLLM/SenseVoice).

## Usage

- transcribe wav file

```bash
$ python main.py
```

![](images/screenshot.png)

- transcribe from microphone

```bash
$ python realtime.py
```

- transcribe from websocket

A basic WebSocket service built with [`Recorder`](https://github.com/xiangyuecn/Recorder) and `FastAPI`; the frontend uses `MP3` format to transmit audio information to reduce latency and increase stability.

```bash
pip install -r requirements-ws-demo.txt
python realtime_ws_server_demo.py
# or run unified HTTP + WS server
python server.py --host 0.0.0.0 --port 8000
```

# check cli options
python realtime_ws_server_demo.py --help

## HTTP 转写接口（server.py）

`POST /api/asr/transcribe` 请求体示例：
```json
{
  "audioBase64": "data:audio/wav;base64,...",
  "chunkMs": 200,
  "language": "zh",
  "textnorm": false,
  "beamSize": 1
}
```

返回：
```json
{
  "text": "完整文本",
  "segments": [
    {"id":0,"beginAt":0.0,"endAt":1.23,"text":"片段文本","timestamps":[10,30]}
  ],
  "sampleRate":16000,
  "chunkMs":200
}
```

环境变量/参数：
- `SENSEVOICE_MODEL_PATH` 模型路径/仓库
- `DEVICE` 推理设备（cpu/cuda）
- `FSMN_VAD_MODEL_PATH` VAD 模型路径/仓库
- `SAMPLERATE` 采样率（默认 16000）
- `CHUNK_MS` 默认分块时长（默认 200）
