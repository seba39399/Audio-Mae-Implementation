"""
core/model_loader.py
Carga y cacheo del modelo AudioMAE pre-entrenado.
"""

import os
import sys
import pathlib
import torch
import streamlit as st


def _patch_repo(repo_dir: str):
    """
    Aplica los parches de compatibilidad al repositorio de AudioMAE:
        - Elimina import de SwinTransformerBlock (no disponible en timm 0.4.12 nuevo)
        - Reemplaza torch._six por math.inf
        - Corrige np.float deprecado en NumPy ≥ 1.24
        - Corrige .cuda() hardcodeado → .to(device)
    """
    import re

    patches = {
        "models_mae.py": [
            (
                "from timm.models.swin_transformer import SwinTransformerBlock",
                "# from timm.models.swin_transformer import SwinTransformerBlock  # patched",
            ),
            (
                "loss_contrastive = torch.FloatTensor([0.0]).cuda()",
                "loss_contrastive = torch.FloatTensor([0.0]).to(imgs.device)",
            ),
        ],
        os.path.join("util", "misc.py"): [
            ("from torch._six import inf", "from math import inf"),
        ],
    }

    for rel_path, replacements in patches.items():
        full_path = os.path.join(repo_dir, rel_path)
        if not os.path.exists(full_path):
            continue
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        for old, new in replacements:
            content = content.replace(old, new)
        # qk_scale removal (models_mae only)
        if "models_mae.py" in rel_path:
            content = content.replace(", qk_scale=None", "")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    # pos_embed: np.float → float
    pos_embed_path = os.path.join(repo_dir, "util", "pos_embed.py")
    if os.path.exists(pos_embed_path):
        with open(pos_embed_path, "r", encoding="utf-8") as f:
            c = f.read()
        c = re.sub(r"np\.float\b", "float", c)
        with open(pos_embed_path, "w", encoding="utf-8") as f:
            f.write(c)


@st.cache_resource(show_spinner="Cargando modelo AudioMAE…")
def load_model(checkpoint_path: str, arch: str = "mae_vit_base_patch16"):
    """
    Construye el modelo AudioMAE y carga los pesos pre-entrenados.
    Usa st.cache_resource para que el modelo se cargue una sola vez.

    Args:
        checkpoint_path: ruta al archivo .pth con los pesos.
        arch: nombre de la arquitectura (por defecto ViT-B/16).

    Returns:
        model en modo eval() listo para inferencia, o None si falla.
    """
    # ── 1. Localizar el repo clonado ─────────────────────────────────────────
    # Buscamos AudioMAE en el directorio actual y en directorios padre
    candidate_dirs = [
        os.path.join(os.getcwd(), "AudioMAE"),
        os.path.join(os.path.dirname(__file__), "..", "AudioMAE"),
        "AudioMAE",
    ]
    repo_dir = None
    for d in candidate_dirs:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "models_mae.py")):
            repo_dir = os.path.abspath(d)
            break

    if repo_dir is None:
        st.error(
            "❌ No se encontró el repositorio AudioMAE. "
            "Clónalo con:\n```\ngit clone https://github.com/facebookresearch/AudioMAE.git\n```"
        )
        return None

    # ── 2. Parches de compatibilidad ─────────────────────────────────────────
    _patch_repo(repo_dir)

    # ── 3. Añadir repo al path de Python ─────────────────────────────────────
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # ── 4. Parche PosixPath → WindowsPath (necesario en Windows) ─────────────
    pathlib.PosixPath = pathlib.WindowsPath

    # ── 5. Importar y construir el modelo ────────────────────────────────────
    try:
        import models_mae
    except ImportError as e:
        st.error(f"❌ No se pudo importar models_mae: {e}")
        return None

    from core.preprocessing import TARGET_LEN, MELBINS

    model = getattr(models_mae, arch)(
        in_chans=1,
        audio_exp=True,
        img_size=(TARGET_LEN, MELBINS),
    )

    # ── 6. Cargar pesos ──────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        st.error(
            f"❌ Checkpoint no encontrado en `{checkpoint_path}`. "
            "Descárgalo desde [Google Drive](https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu) "
            f"y guárdalo en `{checkpoint_path}`."
        )
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
