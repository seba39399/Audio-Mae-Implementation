"""
ui/page_reconstruction.py
Página de la tarea de RECONSTRUCCIÓN / GENERACIÓN.
"""

import os
import tempfile
import urllib.request

import streamlit as st
import torch

from core.preprocessing import load_and_convert, normalize, to_model_input
from core.model_loader import load_model, get_device
from core.reconstruction import run_reconstruction
from utils.visualization import plot_reconstruction_grid, plot_spectrogram

# Audio de demo (ESC-50 — licencia CC)
DEMO_URL = "https://github.com/karolpiczak/ESC-50/raw/master/audio/1-100038-A-14.wav"
DEMO_NAME = "sample_dog_bark.wav"


def _get_demo_audio() -> str:
    """Descarga el audio de demo y devuelve la ruta temporal."""
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, DEMO_NAME)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        with st.spinner("Descargando audio de demo (ESC-50)…"):
            urllib.request.urlretrieve(DEMO_URL, path)
    return path


def render_reconstruction_page(config: dict):
    st.title("🔄 Reconstrucción de Audio — Tarea de Generación")
    st.markdown(
        """
        El modelo AudioMAE **enmascara parches** del mel-spectrogram y luego 
        los **reconstruye** desde cero usando el mecanismo de atención del decoder.  
        Esta es la tarea de *generación*: el modelo aprende a completar información faltante.
        """
    )

    # ── Fuente de audio ───────────────────────────────────────────────────────
    st.subheader("1️⃣ Seleccionar Audio")
    audio_source = st.radio(
        "Fuente de audio:",
        ["🎧 Audio de demo (perro ladrando — ESC-50)", "📁 Subir mi propio archivo .wav"],
        horizontal=True,
    )

    audio_path = None

    if audio_source.startswith("🎧"):
        audio_path = _get_demo_audio()
        st.success(f"Audio de demo listo: `{DEMO_NAME}`")
        st.audio(audio_path)

    else:
        uploaded = st.file_uploader("Sube un archivo .wav", type=["wav"])
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(uploaded.read())
            tmp.close()
            audio_path = tmp.name
            st.audio(audio_path)

    if audio_path is None:
        st.info("⬆️ Sube un archivo de audio para continuar.")
        return

    # ── Preprocesado ─────────────────────────────────────────────────────────
    st.subheader("2️⃣ Preprocesado → Mel-Spectrogram")
    with st.spinner("Calculando mel-spectrogram…"):
        try:
            fbank = load_and_convert(audio_path)
            fbank_norm = normalize(fbank)
        except Exception as e:
            st.error(f"Error al procesar el audio: {e}")
            return

    st.success(
        f"✅ Spectrogram: **{fbank.shape[0]} frames × {fbank.shape[1]} bandas mel**  "
        f"(≈ {fbank.shape[0] * 0.01:.1f} s de audio)"
    )

    with st.expander("Ver mel-spectrogram original", expanded=True):
        fig = plot_spectrogram(fbank_norm, title="Mel-Spectrogram Normalizado (entrada al modelo)")
        st.pyplot(fig)

    # ── Carga del modelo ──────────────────────────────────────────────────────
    st.subheader("3️⃣ Cargar Modelo AudioMAE")
    model = load_model(config["checkpoint_path"])
    if model is None:
        return

    device = get_device()
    st.success(f"✅ Modelo cargado — corriendo en **{device}**")

    total_params = sum(p.numel() for p in model.parameters())
    st.caption(f"Parámetros totales: {total_params/1e6:.1f}M")

    # ── Inferencia ────────────────────────────────────────────────────────────
    st.subheader("4️⃣ Inferencia de Reconstrucción")

    col1, col2 = st.columns([2, 1])
    with col1:
        mask_ratio = st.slider(
            "Mask Ratio",
            0.05, 0.90, config["mask_ratio"], 0.05,
            help="Fracción de parches que el decoder debe reconstruir",
            key="recon_mask_ratio",
        )
    with col2:
        st.metric("Parches ocultos aprox.", f"{mask_ratio*100:.0f}%")

    run_btn = st.button("▶️ Ejecutar Reconstrucción", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Ejecutando inferencia…"):
            result = run_reconstruction(model, fbank_norm, mask_ratio, device)

        # ── Métricas ─────────────────────────────────────────────────────────
        st.subheader("5️⃣ Métricas")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loss (MSE ocultos)", f"{result.loss:.5f}")
        c2.metric("MSE Global", f"{result.mse_global:.5f}")
        c3.metric("SNR aprox.", f"{result.snr_db:.1f} dB")
        c4.metric("% real enmascarado", f"{result.mask_pct:.1f}%")

        # ── Visualización ─────────────────────────────────────────────────────
        st.subheader("6️⃣ Visualización")

        # Rango de frames a mostrar
        max_t = result.original.shape[0]
        col_s, col_e = st.columns(2)
        frame_start = col_s.number_input("Frame inicio", 0, max_t - 10, min(200, max_t // 4))
        frame_end = col_e.number_input("Frame fin", frame_start + 10, max_t, min(800, max_t))

        fig = plot_reconstruction_grid(
            result.original,
            result.masked,
            result.combined,
            result.reconstructed,
            mask_ratio=mask_ratio,
            frame_start=int(frame_start),
            frame_end=int(frame_end),
        )
        st.pyplot(fig)

        # ── Explicación de los paneles ────────────────────────────────────────
        with st.expander("📖 ¿Qué muestra cada panel?"):
            st.markdown(
                """
                | Panel | Descripción |
                |-------|------------|
                | **(a) Original** | Mel-spectrogram real del audio de entrada |
                | **(b) Enmascarado** | Parches ocultados al encoder (zona neutra = valor mínimo) |
                | **(c) Combinación** | Parches visibles originales + parches reconstruidos por el decoder |
                | **(d) Solo reconstruido** | Salida pura del decoder (partes que el modelo generó) |

                Los **parches** son bloques de 16×16 píxeles del espectrogram.  
                El **decoder** recibe los embeddings latentes de los parches visibles + tokens de máscara,
                y predice el valor de cada parche oculto vía atención cruzada.
                """
            )
