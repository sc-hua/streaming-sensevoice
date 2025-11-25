"""简化的 FSMN VAD 迭代器，支持 is_final 标记。"""

from typing import Iterable, Tuple

import numpy as np
from funasr import AutoModel


fsmn_models = {}


class FSMNVADIterator:
    def __init__(self, model_path="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", chunk_size_ms=200, sample_rate=16000, device="cpu", max_preroll_ms=0):
        self.model = self.load_model(model_path, device)
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.cache = {}
        self.in_speech = False
        self.max_preroll_ms = max_preroll_ms
        self.max_preroll_samples = int(max_preroll_ms * sample_rate / 1000)
        self.preroll_buffer = []
        self.preroll_buffer_size = 0

    @staticmethod
    def load_model(model_path: str, device: str):
        key = f"{model_path}-{device}"
        if key not in fsmn_models:
            model = AutoModel(
                model=model_path,
                disable_pbar=True,
                disable_update=True,
                device=device,
                trust_remote_code=True,
            )
            fsmn_models[key] = model
        return fsmn_models[key]

    def reset(self):
        self.cache = {}
        self.in_speech = False
        self.preroll_buffer = []
        self.preroll_buffer_size = 0

    def __call__(self, audio_chunk: np.ndarray, is_final: bool = False) -> Iterable[Tuple[dict, np.ndarray]]:
        # audio_chunk: numpy array (float32)
        
        # Buffer management for PreRoll
        if self.max_preroll_ms > 0 and not self.in_speech:
            self.preroll_buffer.append(audio_chunk)
            self.preroll_buffer_size += len(audio_chunk)
            while self.preroll_buffer_size > self.max_preroll_samples:
                if self.preroll_buffer_size - len(self.preroll_buffer[0]) >= self.max_preroll_samples:
                    removed = self.preroll_buffer.pop(0)
                    self.preroll_buffer_size -= len(removed)
                else:
                    # Slice the first chunk
                    excess = self.preroll_buffer_size - self.max_preroll_samples
                    self.preroll_buffer[0] = self.preroll_buffer[0][excess:]
                    self.preroll_buffer_size -= excess
                    break

        res = self.model.generate(input=audio_chunk, cache=self.cache, is_final=is_final, chunk_size=self.chunk_size_ms)
        value = res[0]["value"]

        speech_dict = {}

        if len(value) > 0:
            for segment in value:
                if segment[1] == -1:
                    speech_dict["start"] = int(segment[0] * self.sample_rate / 1000)
                    self.in_speech = True
                if segment[0] == -1:
                    speech_dict["end"] = int(segment[1] * self.sample_rate / 1000)
                    self.in_speech = False

        if self.in_speech or "end" in speech_dict:
            if "start" in speech_dict and self.max_preroll_ms > 0 and self.preroll_buffer:
                # Prepend preroll
                concatenated = np.concatenate(self.preroll_buffer)
                prepended_samples = len(concatenated) - len(audio_chunk)
                speech_dict["start"] -= prepended_samples
                if speech_dict["start"] < 0:
                    speech_dict["start"] = 0
                audio_chunk = concatenated
                
                # Clear buffer
                self.preroll_buffer = []
                self.preroll_buffer_size = 0

            yield speech_dict, audio_chunk
        
        if not self.in_speech and "end" in speech_dict:
             self.preroll_buffer = []
             self.preroll_buffer_size = 0
