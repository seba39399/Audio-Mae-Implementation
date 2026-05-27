"""
ui/sidebar.py
Barra lateral con navegación y configuración global.
"""

import streamlit as st


def render_sidebar():
    """Renderiza el sidebar y devuelve (página_seleccionada, config_dict)."""

    st.sidebar.image(
        "https://raw.githubusercontent.com/facebookresearch/AudioMAE/main/misc/teaser.png",
        use_column_width=True,
        caption="AudioMAE – NeurIPS 2022",
        output_format="auto",
    ) if False else None  # imagen opcional, descomenta si tienes acceso

    st.sidebar.title("🎵 AudioMAE Demo")
    st.sidebar.markdown(
        """
        **Masked Autoencoders that Listen**  
        [Paper](https://arxiv.org/abs/2207.06405) · [Repo](https://github.com/facebookresearch/AudioMAE)
        """
    )
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navegar a:",
        [
            "🔄 Reconstrucción (Generación)",
            "🏷️ Clasificación",
            "🧠 Arquitectura del Modelo",
        ],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Configuración")

    mask_ratio = st.sidebar.slider(
        "Mask Ratio (fracción de parches ocultos)",
        min_value=0.05,
        max_value=0.90,
        value=0.75,
        step=0.05,
        help="Porcentaje del espectrograma que el modelo debe reconstruir",
    )

    checkpoint_path = st.sidebar.text_input(
        "Ruta al checkpoint (.pth)",
        value="AudioMAE/ckpt/pretrained.pth",
        help="Ruta local al archivo de pesos pre-entrenados de AudioMAE",
    )

    st.sidebar.divider()
    st.sidebar.info(
        "⚠️ Requiere: PyTorch, torchaudio, timm==0.4.12, librosa, soundfile.\n\n"
        "Descarga los pesos desde: [Google Drive](https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu)"
    )

    config = {
        "mask_ratio": mask_ratio,
        "checkpoint_path": checkpoint_path,
    }

    return page, config
