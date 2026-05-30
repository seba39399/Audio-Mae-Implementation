"""
core/classification.py
Tarea de CLASIFICACIÓN con AudioMAE.

Soporta dos modos:
  1. FINE-TUNED (recomendado): carga el checkpoint fine-tuned en AudioSet (527 clases).
     La cabeza de clasificación viene incluida en los pesos → predicción real.
  2. PROTOTIPO (fallback): similitud coseno contra embeddings sintéticos por clase.
     Usado si no hay checkpoint fine-tuned disponible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from core.audioset_labels import AUDIOSET_LABELS

NUM_CLASSES   = 527
EMBEDDING_DIM = 768
TARGET_LEN    = 1024
MELBINS       = 128

DEFAULT_CLASSES = [
    "Speech", "Dog", "Cat", "Guitar", "Piano",
    "Violin, fiddle", "Drum", "Bird vocalization, bird call, bird song",
    "Rain", "Thunderstorm", "Fire", "Car", "Train", "Helicopter",
    "Crowd", "Laughter", "Crying, sobbing", "Clapping", "Singing", "Wind",
]


# ─────────────────────────────────────────────────────────────────────────────
# Cabeza de clasificación para el modelo fine-tuned
# ─────────────────────────────────────────────────────────────────────────────

class AudioClassifier(nn.Module):
    """
    Cabeza de clasificación que se coloca sobre el encoder de AudioMAE.
    Replica la arquitectura usada en el fine-tuning oficial:
      - Mean pooling sobre los tokens de parche (excluye CLS)
      - Capa Linear(768, 527)
    Los pesos vienen del checkpoint fine-tuned.
    """
    def __init__(self, embed_dim: int = EMBEDDING_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (B, N_patches+1, 768)
        x = latent[:, 1:, :].mean(dim=1)   # mean pool sobre parches, excluye CLS
        return self.head(x)                 # (B, 527)


# ─────────────────────────────────────────────────────────────────────────────
# Carga del modelo fine-tuned
# ─────────────────────────────────────────────────────────────────────────────

def load_finetuned_head(checkpoint_path: str, device: torch.device) -> Optional[AudioClassifier]:
    """
    Carga la cabeza de clasificación desde el checkpoint fine-tuned.
    El checkpoint contiene tanto el encoder como la cabeza (key 'head.weight').

    Returns:
        AudioClassifier con pesos cargados, o None si no se puede cargar.
    """
    import os
    if not os.path.exists(checkpoint_path):
        return None

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        # El checkpoint guarda la cabeza como "head.weight" / "head.bias"
        # pero AudioClassifier espera "head.weight" bajo self.head → mapear a eso
        head_state = {}
        for k, v in state.items():
            if k.startswith("head."):
                # "head.weight" → "head.weight"  (ya correcto para self.head)
                new_key = k  # mantener tal cual
                head_state[new_key] = v

        # Fallback: si las claves son solo "weight"/"bias" sin prefijo
        if not head_state:
            for k, v in state.items():
                if k in ("weight", "bias"):
                    head_state[f"head.{k}"] = v

        if not head_state:
            print("[classification] No se encontraron claves de cabeza en el checkpoint.")
            return None

        classifier = AudioClassifier(EMBEDDING_DIM, NUM_CLASSES)
        msg = classifier.load_state_dict(head_state, strict=False)
        print(f"[classification] Cabeza cargada. Missing: {msg.missing_keys}, Unexpected: {msg.unexpected_keys}")
        classifier.to(device)
        classifier.eval()
        return classifier

    except Exception as e:
        print(f"[classification] No se pudo cargar cabeza fine-tuned: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: prototipos sintéticos por similitud coseno
# ─────────────────────────────────────────────────────────────────────────────

def _make_prototype(class_idx: int) -> torch.Tensor:
    """Genera un mel-spectrogram sintético distintivo por clase."""
    rng = np.random.RandomState(seed=class_idx * 137 + 42)
    t = np.arange(TARGET_LEN)
    f = np.arange(MELBINS)

    freq_center = 10 + (class_idx * 13) % (MELBINS - 20)
    freq_width  = 8 + (class_idx * 7) % 30
    burst_period = 30 + (class_idx * 17) % 200
    burst_duty   = 0.2 + (class_idx * 0.07) % 0.7
    mod_freq     = 0.5 + (class_idx * 1.3) % 12.0
    noise_level  = 0.1 + (class_idx * 0.09) % 0.7

    freq_env  = np.exp(-0.5 * ((f - freq_center) / freq_width) ** 2)
    time_env  = (np.sin(2 * np.pi * t / burst_period) > (1 - 2 * burst_duty)).astype(float)
    mod       = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t / TARGET_LEN)
    spec      = np.outer(time_env * mod, freq_env)
    spec     += noise_level * rng.rand(TARGET_LEN, MELBINS)
    spec      = (spec - spec.mean()) / (spec.std() + 1e-8)

    return torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)


def _extract_embedding(model, x: torch.Tensor, device) -> torch.Tensor:
    x = x.to(device)
    with torch.no_grad():
        out = model.forward_encoder(x, mask_ratio=0.0)
        latent = out[0] if isinstance(out, tuple) else out
        emb = latent[:, 1:, :].mean(dim=1)
        emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    embedding: torch.Tensor
    probabilities: torch.Tensor
    class_names: List[str]
    top_k: List[dict]           # [{"class": str, "prob": float}, ...]
    top_class: str
    top_prob: float
    embedding_norm: float
    mode: str                   # "finetuned" o "prototype"


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def run_classification(
    model,
    fbank_norm: torch.Tensor,
    class_names: List[str] = None,
    device: torch.device = None,
    finetuned_ckpt: str = None,
) -> ClassificationResult:
    """
    Clasifica el audio.

    Si se provee finetuned_ckpt y contiene una cabeza de clasificación,
    usa predicción real (527 clases de AudioSet).
    De lo contrario usa similitud coseno con prototipos sintéticos.

    Args:
        model:          encoder AudioMAE en eval().
        fbank_norm:     tensor (TARGET_LEN, MELBINS) normalizado.
        class_names:    clases a mostrar (solo en modo prototipo).
        device:         dispositivo torch.
        finetuned_ckpt: ruta al checkpoint fine-tuned.
    """
    if device is None:
        device = next(model.parameters()).device

    x_input = fbank_norm.unsqueeze(0).unsqueeze(0).float()

    # ── Extraer embedding del audio ───────────────────────────────────────────
    audio_emb = _extract_embedding(model, x_input, device)   # (1, 768)

    # ── Modo 1: Fine-tuned ────────────────────────────────────────────────────
    if finetuned_ckpt:
        head = load_finetuned_head(finetuned_ckpt, device)
        if head is not None:
            x_dev = x_input.to(device)
            with torch.no_grad():
                out = model.forward_encoder(x_dev, mask_ratio=0.0)
                latent = out[0] if isinstance(out, tuple) else out
                logits = head(latent)[0]                     # (527,)
                probs  = torch.sigmoid(logits).cpu()         # multi-label → sigmoid

            # Top-10 clases
            top_vals, top_idxs = probs.topk(10)
            top_k = [
                {"class": AUDIOSET_LABELS[i], "prob": float(v) * 100}
                for v, i in zip(top_vals, top_idxs)
            ]

            return ClassificationResult(
                embedding=audio_emb[0],
                probabilities=probs,
                class_names=AUDIOSET_LABELS,
                top_k=top_k,
                top_class=top_k[0]["class"],
                top_prob=top_k[0]["prob"],
                embedding_norm=float(audio_emb.norm()),
                mode="finetuned",
            )

    # ── Modo 2: Prototipo (fallback) ──────────────────────────────────────────
    if class_names is None:
        class_names = DEFAULT_CLASSES

    proto_embs = []
    for i in range(len(class_names)):
        proto = _make_prototype(i)
        emb   = _extract_embedding(model, proto, device)
        proto_embs.append(emb)

    proto_matrix = torch.cat(proto_embs, dim=0)             # (C, 768)
    similarities = (proto_matrix @ audio_emb.T).squeeze(-1) # (C,)
    probs = F.softmax(similarities / 0.1, dim=0)

    top_vals, top_idxs = probs.topk(min(5, len(class_names)))
    top_k = [
        {"class": class_names[i], "prob": float(v) * 100}
        for v, i in zip(top_vals, top_idxs)
    ]

    return ClassificationResult(
        embedding=audio_emb[0],
        probabilities=probs,
        class_names=class_names,
        top_k=top_k,
        top_class=top_k[0]["class"],
        top_prob=top_k[0]["prob"],
        embedding_norm=float(audio_emb.norm()),
        mode="prototype",
    )
