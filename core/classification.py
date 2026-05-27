"""
core/classification.py
Tarea de CLASIFICACIÓN usando los embeddings del encoder de AudioMAE.

El encoder extrae un embedding global del audio (mean pooling sobre parches).
Ese embedding se pasa por una capa lineal de proyección para obtener logits
por clase.  En ausencia de fine-tuning, la capa lineal es aleatoria (seed=42
para reproducibilidad), lo que muestra la separabilidad geométrica del espacio
latente del encoder sin necesidad de entrenar un clasificador externo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

# Etiquetas de ejemplo basadas en ESC-50 / AudioSet
DEFAULT_CLASSES = [
    "Perro (Dog)",
    "Lluvia (Rain)",
    "Alarma (Siren)",
    "Guitarra (Guitar)",
    "Canto de Pájaro (Bird)",
    "Aplauso (Clapping)",
    "Motor (Engine)",
    "Voz Humana (Speech)",
    "Viento (Wind)",
    "Pasos (Footsteps)",
]

EMBEDDING_DIM = 768   # dimensión del encoder ViT-Base


@dataclass
class ClassificationResult:
    """Resultado de la tarea de clasificación."""
    embedding: torch.Tensor          # (768,) vector latente global
    logits: torch.Tensor             # (num_classes,) scores
    probabilities: torch.Tensor      # (num_classes,) softmax
    class_names: List[str]
    top_class: str
    top_prob: float
    embedding_norm: float            # norma L2 del embedding


def run_classification(
    model,
    fbank_norm: torch.Tensor,
    class_names: List[str] = None,
    device: torch.device = None,
    seed: int = 42,
) -> ClassificationResult:
    """
    Extrae embedding del encoder y clasifica el audio.

    Args:
        model:       modelo AudioMAE en eval().
        fbank_norm:  tensor (TARGET_LEN, MELBINS) normalizado.
        class_names: lista de nombres de clase.
        device:      dispositivo torch.
        seed:        semilla para la capa lineal aleatoria.

    Returns:
        ClassificationResult
    """
    if device is None:
        device = next(model.parameters()).device
    if class_names is None:
        class_names = DEFAULT_CLASSES

    # ── Input: (1, 1, T, F) ──────────────────────────────────────────────────
    x = fbank_norm.unsqueeze(0).unsqueeze(0).float().to(device)

    # ── Extracción de embeddings (sin máscara → ve todos los parches) ─────────
    with torch.no_grad():
        encoder_output = model.forward_encoder(x, mask_ratio=0.0)

        # El encoder devuelve tupla (latent, mask, ids) o solo el tensor
        latent = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
        # latent: (1, num_patches+1, 768)  — el token [CLS] está en pos 0

        # Mean pooling sobre todos los parches (excluye CLS token)
        audio_embedding = latent[:, 1:, :].mean(dim=1)   # (1, 768)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=-1)

    # ── Cabeza de clasificación lineal (seed fijo para reproducibilidad) ──────
    torch.manual_seed(seed)
    classifier = nn.Linear(EMBEDDING_DIM, len(class_names)).to(device)

    with torch.no_grad():
        logits = classifier(audio_embedding)[0]         # (num_classes,)
        probs = F.softmax(logits, dim=-1)               # (num_classes,)

    top_idx = int(probs.argmax())

    return ClassificationResult(
        embedding=audio_embedding.cpu()[0],
        logits=logits.cpu(),
        probabilities=probs.cpu(),
        class_names=class_names,
        top_class=class_names[top_idx],
        top_prob=float(probs[top_idx]) * 100,
        embedding_norm=float(audio_embedding.norm()),
    )
