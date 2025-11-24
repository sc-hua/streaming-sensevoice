"""
统一的 SenseVoice HTTP + WebSocket 服务。

- HTTP: POST /api/asr/transcribe  输入 audioBase64，可选 chunkMs/language/textnorm/beamSize
       返回分段转写结果
- WS  : /api/realtime/ws          复用原 demo，推送 VADEvent + TranscriptionResponse
"""

import base64
import io
import uuid
from typing import Dict, List
from urllib.parse import parse_qs

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from fsmn_vad import FSMNVADIterator
from streaming_sensevoice import StreamingSenseVoice


class Config(BaseSettings, cli_parse_args=True, cli_use_class_docs_for_groups=True):
    HOST: str = Field("0.0.0.0", description="服务监听 Host")
    PORT: int = Field(8000, description="服务监听端口")
    DEBUG: bool = Field(False, description="Debug 模式")

    SENSEVOICE_MODEL_PATH: str = Field(
        "iic/SenseVoiceSmall", 
        description="SenseVoice 模型路径/仓库")
    FSMN_VAD_MODEL_PATH: str = Field(
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
        description="FSMN-VAD 模型路径/仓库"
    )
    DEVICE: str = Field("cpu", description="推理设备，cpu/cuda")
    SAMPLERATE: int = Field(16000, description="采样率")
    CHUNK_MS: int = Field(200, description="分块时长（毫秒）")
    BEAM_SIZE: int = Field(1, description="CTC beam size（>1 需提供上下文）")
    LANGUAGE: str = Field("zh", description="语言，auto/zh/en/ja/ko/yue")
    TEXTNORM: bool = Field(False, description="是否启用逆文本规范化")

    class Config:
        extra = "ignore"


config = Config()

# 预加载模型
StreamingSenseVoice.load_model(model=config.SENSEVOICE_MODEL_PATH, device=config.DEVICE)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscribeRequest(BaseModel):
    audioBase64: str
    chunkMs: int | None = None
    language: str | None = None
    textnorm: bool | None = None
    beamSize: int | None = None


def _decode_base64(audio_base64: str) -> bytes:
    if "," in audio_base64:
        audio_base64 = audio_base64.split(",", 1)[1]
    return base64.b64decode(audio_base64)


def _load_audio(audio_bytes: bytes, expected_sr: int) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != expected_sr:
        raise HTTPException(status_code=400, detail=f"采样率需为 {expected_sr}Hz，当前为 {sr}Hz")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32)


def _run_transcription(
    samples: np.ndarray,
    chunk_ms: int,
    language: str,
    textnorm: bool,
    beam_size: int,
) -> Dict:
    model = StreamingSenseVoice(
        model=config.SENSEVOICE_MODEL_PATH,
        device=config.DEVICE,
        language=language,
        textnorm=textnorm,
        beam_size=beam_size,
    )
    vad = FSMNVADIterator(
        model_path=config.FSMN_VAD_MODEL_PATH,
        chunk_size_ms=chunk_ms,
        sample_rate=config.SAMPLERATE,
    )

    chunk_size = max(int(config.SAMPLERATE * chunk_ms / 1000), 1)
    segments: List[Dict] = []
    current_segment = None
    detected = False
    seg_id = 0
    cursor = 0
    total_len = len(samples)

    def finalize(end_sample: int | None = None):
        nonlocal current_segment, detected, seg_id
        if current_segment and detected:
            if end_sample is not None:
                current_segment["endAt"] = end_sample / config.SAMPLERATE
            elif "endAt" not in current_segment:
                current_segment["endAt"] = total_len / config.SAMPLERATE
            segments.append(current_segment)
            seg_id += 1
        current_segment = None
        detected = False

    while cursor < total_len:
        chunk = samples[cursor : cursor + chunk_size]
        cursor += chunk_size
        is_final = cursor >= total_len

        for speech_dict, speech_samples in vad(chunk, is_final=is_final):
            if "start" in speech_dict:
                model.reset()
                current_segment = {
                    "id": seg_id,
                    "beginAt": speech_dict["start"] / config.SAMPLERATE,
                    "text": "",
                    "timestamps": [],
                }
                detected = False

            is_last = "end" in speech_dict
            for res in model.streaming_inference(speech_samples, is_last):
                text = res.get("text") or ""
                if text:
                    detected = True
                    if current_segment is None:
                        current_segment = {
                            "id": seg_id,
                            "beginAt": cursor / config.SAMPLERATE,
                            "text": "",
                            "timestamps": [],
                        }
                    current_segment["text"] += text
                    current_segment["timestamps"] = res.get("timestamps") or []

            if is_last:
                finalize(speech_dict.get("end"))

    if current_segment and detected:
        finalize()

    return {
        "text": "".join(seg.get("text", "") for seg in segments),
        "segments": segments,
        "sampleRate": config.SAMPLERATE,
        "chunkMs": chunk_ms,
    }


@app.post("/api/asr/transcribe")
async def transcribe(body: TranscribeRequest = Body(...)):
    if not body.audioBase64:
        raise HTTPException(status_code=400, detail="audioBase64 不能为空")
    try:
        audio_bytes = _decode_base64(body.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="audioBase64 解码失败")

    samples = _load_audio(audio_bytes, expected_sr=config.SAMPLERATE)
    chunk_ms = int(body.chunkMs or config.CHUNK_MS)
    language = body.language or config.LANGUAGE
    textnorm = body.textnorm if body.textnorm is not None else config.TEXTNORM
    beam_size = int(body.beamSize or config.BEAM_SIZE)

    return _run_transcription(
        samples=samples,
        chunk_ms=chunk_ms,
        language=language,
        textnorm=textnorm,
        beam_size=beam_size,
    )


class TranscriptionChunk(BaseModel):
    timestamps: list[int]
    raw_text: str
    final_text: str | None = None
    spk_id: int | None = None


class TranscriptionResponse(BaseModel):
    type: str = "TranscriptionResponse"
    id: int
    begin_at: float
    end_at: float | None
    data: TranscriptionChunk
    is_final: bool
    session_id: str | None = None


class VADEvent(BaseModel):
    type: str = "VADEvent"
    is_active: bool


@app.websocket("/api/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()

        session_id = str(uuid.uuid4())

        query_params = parse_qs(websocket.scope.get("query_string", b"").decode())
        chunk_duration = float(query_params.get("chunk_duration", config.CHUNK_MS / 1000.0))

        sensevoice_model = StreamingSenseVoice(
            model=config.SENSEVOICE_MODEL_PATH, device=config.DEVICE
        )

        vad_iterator = FSMNVADIterator(
            model_path=config.FSMN_VAD_MODEL_PATH,
            chunk_size_ms=int(chunk_duration * 1000),
            sample_rate=config.SAMPLERATE,
        )

        audio_buffer = np.array([], dtype=np.float32)
        chunk_size = int(chunk_duration * config.SAMPLERATE)

        speech_count = 0
        current_audio_begin_time = 0.0

        asr_detected = False

        transcription_response: TranscriptionResponse | None = None
        while True:
            data = await websocket.receive_bytes()

            buffer = io.BytesIO(data)
            try:
                buffer.name = "a.mp3"
                samples, sr = sf.read(buffer, dtype="float32")
                audio_buffer = np.concatenate((audio_buffer, samples))
            except Exception:
                continue
            finally:
                buffer.close()

            if sr != config.SAMPLERATE:
                await websocket.send_json({"type": "error", "message": f"采样率需为 {config.SAMPLERATE}Hz，当前 {sr}Hz"})
                continue

            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]

                for speech_dict, speech_samples in vad_iterator(chunk):
                    if "start" in speech_dict:
                        sensevoice_model.reset()

                        current_audio_begin_time = speech_dict["start"] / config.SAMPLERATE

                        if asr_detected:
                            speech_count += 1
                        asr_detected = False

                        await websocket.send_json(VADEvent(is_active=True).model_dump())

                    is_last = "end" in speech_dict

                    for res in sensevoice_model.streaming_inference(
                        speech_samples, is_last
                    ):

                        if len(res.get("text", "")) > 0:
                            asr_detected = True

                        if asr_detected:
                            transcription_response = TranscriptionResponse(
                                id=speech_count,
                                begin_at=current_audio_begin_time,
                                end_at=None,
                                data=TranscriptionChunk(
                                    timestamps=res.get("timestamps", []), raw_text=res.get("text", "")
                                ),
                                is_final=False,
                                session_id=session_id,
                            )
                            await websocket.send_json(
                                transcription_response.model_dump()
                            )

                    if is_last:
                        if asr_detected and transcription_response:
                            speech_count += 1
                            asr_detected = False

                            transcription_response.is_final = True
                            transcription_response.end_at = (
                                speech_dict["end"] / config.SAMPLERATE
                            )

                            await websocket.send_json(
                                transcription_response.model_dump()
                            )
                        await websocket.send_json(
                            VADEvent(is_active=False).model_dump()
                        )

    except WebSocketDisconnect:
        pass
    finally:
        try:
            sensevoice_model.reset()
            del sensevoice_model
            del vad_iterator
            del audio_buffer
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
