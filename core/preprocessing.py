"""
core/preprocessing.py
Pipeline de preprocesado de audio → mel-spectrogram.
Parámetros extraídos del paper AudioMAE (NeurIPS 2022).
"""

import torch
import torchaudio
import soundfile as sf
import numpy as np

# ── Constantes del paper ──────────────────────────────────────────────────────
MELBINS = 128        # bandas mel
TARGET_LEN = 1024    # frames de tiempo (10ms x 1024 ≈ 10 s a 16 kHz)
NORM_MEAN = -4.2677393
NORM_STD = 4.5689974


def load_and_convert(filepath: str) -> torch.Tensor:
    """
    Carga un archivo .wav y lo convierte a mel-spectrogram (Kaldi FBANK).

    Pasos:
        1. Lectura con soundfile (evita problemas de backend en Windows/Linux).
        2. Conversión a mono si es estéreo.
        3. Normalización DC (elimina offset de CC).
        4. Repetición del audio si es < 10 s (evita zero-padding que degrada la reconstrucción).
        5. Escala a rango 16-bit PCM para compatibilidad con kaldi.fbank.
        6. Cálculo del FBANK con parámetros del paper.
        7. Pad/crop para garantizar exactamente TARGET_LEN frames.

    Args:
        filepath: ruta al archivo de audio .wav

    Returns:
        fbank: Tensor de forma (TARGET_LEN, MELBINS) — sin normalizar
    """
    # 1. Lectura nativa
    data, samplerate = sf.read(filepath)

    # 2. Tensor (canales, muestras)
    waveform = torch.FloatTensor(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)          # (1, N)
    else:
        waveform = waveform.t()                   # (C, N)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3. Normalización DC
    waveform = waveform - waveform.mean()

    # 4. Repetición para audios cortos (< ~10 s)
    target_samples = int(TARGET_LEN * 0.01 * samplerate)
    actual_samples = waveform.shape[1]
    if actual_samples < target_samples:
        repeats = int(target_samples / actual_samples) + 1
        waveform = waveform.repeat(1, repeats)[:, :target_samples]

    # 5. Escala PCM
    waveform = waveform * 32768.0

    # 6. Kaldi FBANK  (mismo que el paper)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=samplerate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=MELBINS,
        dither=0.0,
        frame_shift=10,
    )

    # 7. Pad / crop
    n = fbank.shape[0]
    if n < TARGET_LEN:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, TARGET_LEN - n))(fbank)
    else:
        fbank = fbank[:TARGET_LEN]

    return fbank          # (1024, 128)


def normalize(fbank: torch.Tensor) -> torch.Tensor:
    """Normalización global con media y std del paper."""
    return (fbank - NORM_MEAN) / (NORM_STD * 2)


def denormalize(fbank: torch.Tensor) -> torch.Tensor:
    """Desnormalización para visualización."""
    return fbank * (NORM_STD * 2) + NORM_MEAN


def to_model_input(fbank_norm: torch.Tensor) -> torch.Tensor:
    """
    Convierte el spectrogram normalizado al tensor de entrada del modelo.
    Shape: (1, 1, TARGET_LEN, MELBINS)  →  [batch, canal, tiempo, freq]
    """
    return fbank_norm.unsqueeze(0).unsqueeze(0).float()
