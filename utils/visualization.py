"""
utils/visualization.py
Funciones de visualización de mel-spectrogramas y resultados.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple


def _to_numpy(tensor) -> np.ndarray:
    """Convierte Tensor o ndarray a numpy 2D."""
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)
    # Si viene como (T, F) → queremos imshow en (F, T) con origin='lower'
    # Se asume que la dimensión larga es el eje de tiempo
    if data.ndim == 2 and data.shape[0] > data.shape[1]:
        data = data.T   # (F, T)
    return data


def _color_limits(data: np.ndarray) -> Tuple[float, float]:
    """Límites de color estadísticos (±2σ alrededor de la media)."""
    m, s = data.mean(), data.std()
    vmin, vmax = m - 2 * s, m + 2 * s
    if vmax - vmin < 4.0:
        vmin, vmax = -2.0, 4.0
    return vmin, vmax


def plot_spectrogram(
    fbank,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 3),
    ax=None,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Dibuja un mel-spectrogram individual.

    Args:
        fbank:  Tensor o ndarray (T, F) o (F, T).
        title:  título del subplot.
        vmin/vmax: límites de color (calculados automáticamente si None).
        figsize: tamaño de figura si ax es None.
        ax:     eje matplotlib existente.
        cmap:   colormap.
    """
    data = _to_numpy(fbank)
    if vmin is None or vmax is None:
        vmin, vmax = _color_limits(data)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        data, origin="lower", aspect="auto",
        interpolation="nearest", cmap=cmap,
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Mel band")
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    return fig


def plot_reconstruction_grid(
    original, masked, combined, reconstructed,
    mask_ratio: float = 0.75,
    frame_start: int = 200,
    frame_end: int = 800,
) -> plt.Figure:
    """
    Cuadrícula 4×1 con las 4 vistas de reconstrucción.

    Args:
        original:      (T, F) espectrograma original normalizado.
        masked:        (T, F) con parches ocultos.
        combined:      (T, F) visible + reconstruido.
        reconstructed: (T, F) solo predicción del decoder.
        mask_ratio:    usado en el título.
        frame_start/end: rango de frames de tiempo a mostrar.
    """
    # Segmento de tiempo
    def crop(t):
        data = _to_numpy(t)
        # data es (F, T) después de _to_numpy
        return data[:, frame_start:frame_end]

    orig_c = crop(original)
    mask_c = crop(masked)
    comb_c = crop(combined)
    reco_c = crop(reconstructed)

    vmin, vmax = _color_limits(_to_numpy(original))

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    panels = [
        (orig_c, f"(a) Original  [mask_ratio={mask_ratio}]"),
        (mask_c, "(b) Enmascarado — parches ocultos por el modelo"),
        (comb_c, "(c) Reconstrucción combinada — original + parches predichos"),
        (reco_c, "(d) Solo reconstruido — salida del decoder"),
    ]

    for i, (data, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(
            data, origin="lower", aspect="auto",
            interpolation="nearest", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel band")
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    return fig


def plot_classification_bars(
    class_names, probabilities,
    highlight_top: bool = True,
) -> plt.Figure:
    """
    Gráfico de barras horizontales con probabilidades de clasificación.
    """
    probs_np = probabilities.detach().cpu().numpy() * 100
    colors = ["#e74c3c" if highlight_top and i == probs_np.argmax() else "#3498db"
              for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(8, max(4, len(class_names) * 0.55)))
    bars = ax.barh(class_names, probs_np, color=colors, edgecolor="none", height=0.6)

    for bar, prob in zip(bars, probs_np):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{prob:.1f}%", va="center", fontsize=9,
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Probabilidad (%)")
    ax.set_title("Distribución de Probabilidades por Clase", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_embedding_heatmap(embedding: torch.Tensor) -> plt.Figure:
    """
    Visualiza el vector de embedding como mapa de calor 2D.
    Útil para explicar el espacio latente del encoder.
    """
    emb_np = embedding.detach().cpu().numpy()
    side = int(np.ceil(np.sqrt(len(emb_np))))
    padded = np.zeros(side * side)
    padded[: len(emb_np)] = emb_np
    grid = padded.reshape(side, side)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="RdBu_r", aspect="auto")
    ax.set_title("Embedding Latente (768-dim) del Encoder", fontsize=10)
    ax.set_xlabel("Dimensión →")
    ax.set_ylabel("Dimensión →")
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    return fig
