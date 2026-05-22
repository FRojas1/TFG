"""
ASR model wrappers for benchmarking.

Supported backends:
  - whisper   : OpenAI Whisper via HuggingFace transformers
  - wav2vec2  : Wav2Vec2 / HuBERT CTC models via HuggingFace transformers
  - nemo      : Nvidia NeMo models (Parakeet, Canary) — requires nemo_toolkit[asr]
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class ASRModel(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# --------------------------------------------------------------------------- #
# Whisper (encoder-decoder, seq2seq)
# --------------------------------------------------------------------------- #

class WhisperModel(ASRModel):
    def __init__(self, model_id, device="cuda", dtype=torch.float16):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self._model_id = model_id
        self._name = model_id.split("/")[-1]

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )

    @property
    def name(self):
        return self._name

    def transcribe(self, audio, sample_rate):
        result = self.pipe(
            {"raw": audio, "sampling_rate": sample_rate},
            return_timestamps=True,
        )
        return result["text"].strip()


# --------------------------------------------------------------------------- #
# Wav2Vec2 / HuBERT (CTC, no LM decoder)
# --------------------------------------------------------------------------- #

class Wav2Vec2Model(ASRModel):
    def __init__(self, model_id, device="cuda", dtype=torch.float32):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._model_id = model_id
        self._name = model_id.split("/")[-1]
        self.device = device

        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        self.model.eval()

    @property
    def name(self):
        return self._name

    def transcribe(self, audio, sample_rate):
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0].strip()


# --------------------------------------------------------------------------- #
# FastConformer CTC (Parakeet) via HuggingFace Transformers
# --------------------------------------------------------------------------- #

class FastConformerCTCModel(ASRModel):
    """Wrapper for NVIDIA FastConformer CTC models (e.g. Parakeet-CTC-1.1B)
    using HuggingFace Transformers AutoModelForCTC + AutoProcessor."""

    def __init__(self, model_id, device="cuda", dtype=torch.float16):
        from transformers import AutoModelForCTC, AutoProcessor

        self._model_id = model_id
        self._name = model_id.split("/")[-1]
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self._sr = self.processor.feature_extractor.sampling_rate

    @property
    def name(self):
        return self._name

    def transcribe(self, audio, sample_rate):
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device, dtype=self.model.dtype)
                  if v.dtype.is_floating_point else v.to(self.device)
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        return self.processor.batch_decode(outputs)[0].strip()


# --------------------------------------------------------------------------- #
# NeMo (Canary, other NeMo-only models)
# --------------------------------------------------------------------------- #

class NeMoModel(ASRModel):
    """Wrapper for Nvidia NeMo ASR models.

    Requires: pip install nemo_toolkit[asr]
    """

    def __init__(self, model_name, device="cuda"):
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "nemo_toolkit is required for NeMo models. "
                "Install with: pip install nemo_toolkit[asr]"
            )

        self._name = model_name.split("/")[-1] if "/" in model_name else model_name
        self.device = device
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()

    @property
    def name(self):
        return self._name

    def transcribe(self, audio, sample_rate):
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            result = self.model.transcribe([tmp.name])

        text = result[0]
        if not isinstance(text, str):
            text = text.text
        return text.strip()


# --------------------------------------------------------------------------- #
# Moonshine (encoder-decoder, lightweight seq2seq)
# --------------------------------------------------------------------------- #

class MoonshineModel(ASRModel):
    """Wrapper for Useful Sensors Moonshine models via HuggingFace transformers.

    Implements manual chunking since the HF pipeline doesn't support
    return_timestamps for non-Whisper seq2seq models.
    """

    CHUNK_SECONDS = 15
    OVERLAP_SECONDS = 1

    def __init__(self, model_id, device="cuda", dtype=torch.float16):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self._model_id = model_id
        self._name = model_id.split("/")[-1]
        self.device = device
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()

    @property
    def name(self):
        return self._name

    def _transcribe_chunk(self, audio_chunk, sample_rate):
        inputs = self.processor(
            audio_chunk, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device, dtype=self.dtype)
                  if v.dtype.is_floating_point else v.to(self.device)
                  for k, v in inputs.items()}

        max_tokens = min(448, int(len(audio_chunk) / sample_rate * 12) + 10)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def transcribe(self, audio, sample_rate):
        chunk_samples = self.CHUNK_SECONDS * sample_rate
        overlap_samples = self.OVERLAP_SECONDS * sample_rate
        stride = chunk_samples - overlap_samples

        if len(audio) <= chunk_samples:
            return self._transcribe_chunk(audio, sample_rate)

        parts = []
        offset = 0
        while offset < len(audio):
            end = min(offset + chunk_samples, len(audio))
            chunk = audio[offset:end]
            if len(chunk) > sample_rate * 0.5:
                parts.append(self._transcribe_chunk(chunk, sample_rate))
            offset += stride

        return " ".join(parts)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_BACKENDS = {
    "whisper": WhisperModel,
    "wav2vec2": Wav2Vec2Model,
    "fastconformer_ctc": FastConformerCTCModel,
    "nemo": NeMoModel,
    "moonshine": MoonshineModel,
}


def load_model(config: dict) -> ASRModel:
    """Instantiate an ASR model from a YAML config dict."""
    model_cfg = config["model"]
    inf_cfg = config.get("inference", {})

    backend = model_cfg["backend"]
    model_id = model_cfg["model_id"]
    device = inf_cfg.get("device", "cuda")

    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: {list(_BACKENDS.keys())}"
        )

    cls = _BACKENDS[backend]

    if backend == "nemo":
        return cls(model_id, device=device)

    dtype_name = inf_cfg.get("dtype", "float16")
    dtype = getattr(torch, dtype_name, torch.float16)
    return cls(model_id, device=device, dtype=dtype)
