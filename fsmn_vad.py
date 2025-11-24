from funasr import AutoModel

class FSMNVADIterator:
    def __init__(self, model_path="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", chunk_size_ms=200, sample_rate=16000):
        self.model = AutoModel(model=model_path, disable_pbar=True, disable_update=True)
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.cache = {}
        self.in_speech = False
        
    def __call__(self, audio_chunk):
        # audio_chunk: numpy array (float32)
        
        res = self.model.generate(input=audio_chunk, cache=self.cache, is_final=False, chunk_size=self.chunk_size_ms)
        value = res[0]["value"]
        
        speech_dict = {}
        
        if len(value) > 0:
            for segment in value:
                if segment[1] == -1:
                    # Start detected
                    # segment[0] is in ms
                    speech_dict["start"] = int(segment[0] * self.sample_rate / 1000)
                    self.in_speech = True
                if segment[0] == -1:
                    # End detected
                    # segment[1] is in ms
                    speech_dict["end"] = int(segment[1] * self.sample_rate / 1000)
                    self.in_speech = False
        
        if self.in_speech or "end" in speech_dict:
            yield speech_dict, audio_chunk
