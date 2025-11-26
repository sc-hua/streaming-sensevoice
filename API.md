# API 文档

本项目提供 HTTP 和 WebSocket 两种接口用于语音转写。

## 配置

服务启动时可以通过环境变量或命令行参数进行配置。

| 参数名 | 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--host` | `HOST` | `0.0.0.0` | 服务监听 Host |
| `--port` | `PORT` | `9000` | 服务监听端口 |
| `--debug` | `DEBUG` | `False` | Debug 模式 |
| `--sensevoice_model_path` | `SENSEVOICE_MODEL_PATH` | `ckpts/sensevoice-small` | SenseVoice 模型路径/仓库 |
| `--fsmn_vad_model_path` | `FSMN_VAD_MODEL_PATH` | `ckpts/fsmn-vad-zh-cn-16k` | FSMN-VAD 模型路径/仓库 |
| `--device` | `DEVICE` | `cuda` | 推理设备 (cpu/cuda) |
| `--samplerate` | `SAMPLERATE` | `16000` | 采样率 |
| `--chunk_ms` | `CHUNK_MS` | `200` | 分块时长（毫秒） |
| `--beam_size` | `BEAM_SIZE` | `1` | CTC beam size |
| `--language` | `LANGUAGE` | `zh` | 语言 (auto/zh/en/ja/ko/yue) |
| `--textnorm` | `TEXTNORM` | `False` | 是否启用逆文本规范化 |
| `--preroll_ms` | `PREROLL_MS` | `600` | VAD Pre-roll duration (ms) |

## HTTP 接口

### 健康检查

检查服务是否正常运行。

- **URL**: `/health` 或 `/`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "ok",
    "message": "ASR API Server is running"
  }
  ```

### 语音转写

提交音频数据（Base64编码）进行转写。

- **URL**: `/api/asr/transcribe`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `audioBase64` | string | 是 | 音频文件的 Base64 编码字符串 (支持带 data URI scheme 前缀) |
| `chunkMs` | int | 否 | 分块时长 (ms)，默认使用服务配置 |
| `language` | string | 否 | 语言代码，默认使用服务配置 |
| `textnorm` | boolean | 否 | 是否启用逆文本规范化，默认使用服务配置 |
| `beamSize` | int | 否 | Beam Search 大小，默认使用服务配置 |

**请求示例**:

```json
{
  "audioBase64": "data:audio/wav;base64,UklGRi...",
  "chunkMs": 200,
  "language": "zh",
  "textnorm": false,
  "beamSize": 1
}
```

#### 响应内容

返回完整的转写文本及分段详情。

```json
{
  "text": "完整转写文本内容",
  "segments": [
    {
      "id": 0,
      "beginAt": 0.0,
      "endAt": 2.5,
      "text": "分段文本",
      "timestamps": [ [100, 200], [300, 400] ]
    }
  ],
  "sampleRate": 16000,
  "chunkMs": 200
}
```

## WebSocket 接口

提供流式语音转写功能，支持 VAD（语音活动检测）事件推送。

- **URL**: `/ws/asr/transcribe`
- **Query Parameters**:
  - `chunk_duration`: (可选) 处理分块时长（秒），默认为配置中的 `CHUNK_MS / 1000`。

### 交互流程

1. **建立连接**: 客户端连接 WebSocket。
2. **发送音频**: 客户端持续发送二进制音频数据（支持常见音频格式，如 mp3, wav 等，服务端会自动解码并重采样至 16000Hz）。
3. **接收消息**: 服务端推送 JSON 格式的消息。

### 消息类型

服务端发送的消息包含 `type` 字段，主要有以下几种类型：

#### 1. VAD 事件 (`VADEvent`)

通知语音活动的开始或结束。

```json
{
  "type": "VADEvent",
  "is_active": true  // true 表示语音开始，false 表示语音结束
}
```

#### 2. 转写响应 (`TranscriptionResponse`)

包含实时的转写结果。

```json
{
  "type": "TranscriptionResponse",
  "id": 1,                  // 语音片段序号
  "begin_at": 1.5,          // 该片段在整个流中的开始时间（秒）
  "end_at": null,           // 结束时间（非最终结果为 null）
  "is_final": false,        // 是否为该片段的最终结果
  "session_id": "uuid...",  // 会话 ID
  "data": {
    "timestamps": [],       // 时间戳信息
    "raw_text": "正在转写的文本...",
    "final_text": null,
    "spk_id": null
  }
}
```

当 `is_final` 为 `true` 时，表示该语音片段处理完成，`end_at` 会有具体数值。

#### 3. 错误消息

```json
{
  "type": "error",
  "message": "错误描述信息"
}
```
