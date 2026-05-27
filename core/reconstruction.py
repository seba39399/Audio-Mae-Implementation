"""
core/reconstruction.py
Lógica de inferencia para la tarea de RECONSTRUCCIÓN / GENERACIÓN.

El modelo AudioMAE:
  1. Parchea el espectrograma en bloques (patches) de 16×16 px.
  2. Enmascara un % de esos parches (mask_ratio).
  3. El encoder ViT procesa solo los parches VISIBLES → genera latents.
  4. El decoder reconstruye los parches OCULTOS a partir de los latents.
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class ReconstructionResult:
    """Resultado completo de una inferencia de reconstrucción."""
    original: torch.Tensor       # (T, F) espectrograma normalizado original
    masked: torch.Tensor         # (T, F) con parches ocultos
    reconstructed: torch.Tensor  # (T, F) predicción del decoder
    combined: torch.Tensor       # (T, F) visible original + reconstruido oculto
    mask: torch.Tensor           # (T, F) binaria: 1=oculto, 0=visible
    loss: float                  # MSE en parches ocultos
    mask_pct: float              # porcentaje real enmascarado
    mse_global: float            # MSE entre original y reconstruido
    snr_db: float                # Signal-to-Noise Ratio en dB


def run_reconstruction(
    model,
    fbank_norm: torch.Tensor,
    mask_ratio: float = 0.75,
    device: torch.device = None,
) -> ReconstructionResult:
    """
    Ejecuta la inferencia de reconstrucción con AudioMAE.

    Args:
        model:       modelo AudioMAE cargado y en eval().
        fbank_norm:  tensor (TARGET_LEN, MELBINS) normalizado.
        mask_ratio:  fracción de parches a ocultar [0..1].
        device:      dispositivo torch.

    Returns:
        ReconstructionResult con todos los tensores y métricas.
    """
    if device is None:
        device = next(model.parameters()).device

    # ── Preparar input: (1, 1, T, F) ─────────────────────────────────────────
    x = fbank_norm.unsqueeze(0).unsqueeze(0).float().to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        loss, y_pred, mask, _ = model(x, mask_ratio=mask_ratio)

    # ── Reconstruir espectrograma desde parches ───────────────────────────────
    y = model.unpatchify(y_pred).cpu()        # (1, 1, T, F)

    # ── Expandir máscara a resolución de espectrograma ────────────────────────
    # mask shape: (1, num_patches)  →  (1, 1, T, F)
    p_h, p_w = model.patch_embed.patch_size
    mask_expanded = mask.detach().cpu()
    mask_expanded = mask_expanded.unsqueeze(-1).repeat(1, 1, p_h * p_w)
    mask_expanded = model.unpatchify(mask_expanded)   # 1=oculto, 0=visible

    x_cpu = x.cpu()

    # ── Calcular límites de color estadísticos ────────────────────────────────
    mean_val = float(x_cpu.mean())
    std_val = float(x_cpu.std())
    vmin = mean_val - 2 * std_val
    vmax = mean_val + 2 * std_val
    if vmax - vmin < 4.0:
        vmin, vmax = -2.0, 4.0

    # ── Construir visualizaciones 2D (T, F) ───────────────────────────────────
    original_2d = x_cpu[0, 0]                                    # (T, F)
    masked_2d = x_cpu[0, 0] * (1 - mask_expanded[0, 0]) + mask_expanded[0, 0] * vmin
    combined_2d = x_cpu[0, 0] * (1 - mask_expanded[0, 0]) + y[0, 0] * mask_expanded[0, 0]
    recon_2d = y[0, 0]

    # ── Métricas ──────────────────────────────────────────────────────────────
    mse_global = torch.nn.functional.mse_loss(x_cpu, y).item()

    signal_power = original_2d.pow(2).mean().item()
    noise_power = (original_2d - recon_2d).pow(2).mean().item()
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

    return ReconstructionResult(
        original=original_2d,
        masked=masked_2d,
        reconstructed=recon_2d,
        combined=combined_2d,
        mask=mask_expanded[0, 0],
        loss=loss.item(),
        mask_pct=mask.mean().item() * 100,
        mse_global=mse_global,
        snr_db=snr_db,
    )
