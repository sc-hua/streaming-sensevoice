import asyncio
import base64
import json
import time
import io
import random
import requests
import websockets
import soundfile as sf
import numpy as np

# 配置
HOST = "127.0.0.1"
PORT = 9000
HTTP_URL = f"http://{HOST}:{PORT}"
WS_URL = f"ws://{HOST}:{PORT}/api/realtime/ws"
AUDIO_FILE = "test_16k.wav"

def test_http_transcribe():
    print(f"--- Testing HTTP Transcribe ({HTTP_URL}/api/asr/transcribe) ---")
    
    try:
        with open(AUDIO_FILE, "rb") as f:
            audio_data = f.read()
    except FileNotFoundError:
        print(f"Error: File {AUDIO_FILE} not found.")
        return

    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    payload = {
        "audioBase64": audio_base64,
        "chunkMs": 200,
        "language": "zh",
        "textnorm": False,
        "beamSize": 1
    }

    start_time = time.time()
    try:
        response = requests.post(f"{HTTP_URL}/api/asr/transcribe", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Time Taken: {time.time() - start_time:.4f}s")
        print("Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")

async def test_websocket_realtime():
    print(f"\n--- Testing WebSocket Realtime ({WS_URL}) ---")
    
    try:
        data, sr = sf.read(AUDIO_FILE, dtype="float32")
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return

    # 模拟切片发送
    async with websockets.connect(WS_URL) as websocket:
        print("WebSocket Connected")
        
        async def receive_messages():
            try:
                async for message in websocket:
                    msg = json.loads(message)
                    if msg['type'] == 'TranscriptionResponse':
                        print(f"TRANSCRIPTION: {msg['data']['raw_text']} (Final: {msg['is_final']})")
                    elif msg['type'] == 'VADEvent':
                        print(f"VAD: {'Active' if msg['is_active'] else 'Inactive'}")
                    else:
                        print(f"MSG: {msg}")
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket Connection Closed")

        # 启动接收任务
        recv_task = asyncio.create_task(receive_messages())

        # 发送音频数据
        total_len = len(data)
        cursor = 0
        
        while cursor < total_len:
            # 随机切片时长 0.2-0.8s
            chunk_duration = random.uniform(0.2, 0.8)
            chunk_size = int(sr * chunk_duration)

            chunk = data[cursor : cursor + chunk_size]
            cursor += chunk_size
            
            # 将 chunk 封装为 WAV 格式的 bytes
            # 因为 server 端是 sf.read(io.BytesIO(data))，它期望每一包都是完整音频文件
            buffer = io.BytesIO()
            sf.write(buffer, chunk, sr, format='WAV')
            audio_bytes = buffer.getvalue()
            
            await websocket.send(audio_bytes)
            print(f"Sent chunk: {len(chunk)} samples ({len(audio_bytes)} bytes) duration: {chunk_duration:.2f}s")
            
            # 模拟实时发送间隔
            await asyncio.sleep(chunk_duration)

        # 等待一会儿以接收剩余结果
        await asyncio.sleep(2)
        await websocket.close()
        await recv_task

if __name__ == "__main__":
    # 先测试 HTTP
    test_http_transcribe()
    
    # 再测试 WebSocket
    # 注意：需要 server 正在运行
    try:
        asyncio.run(test_websocket_realtime())
    except KeyboardInterrupt:
        pass
    except ConnectionRefusedError:
        print("\nError: Connection refused. Is the server running?")
