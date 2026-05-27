"""
ui/page_classification.py
Página de la tarea de CLASIFICACIÓN.
"""

import os
import tempfile
import urllib.request

import streamlit as st
import torch

from core.preprocessing import load_and_convert, normalize
from core.model_loader import load_model, get_device
from core.classification import run_classification, DEFAULT_CLASSES
from utils.visualization import plot_classification_bars, plot_embedding_heatmap, plot_spectrogram

DEMO_URL = "https://github.com/karolpiczak/ESC-50/raw/master/audio/1-100038-A-14.wav"
DEMO_NAME = "sample_dog_bark.wav"


def _get_demo_audio() -> str:
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, DEMO_NAME)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        with st.spinner("Descargando audio de demo…"):
            urllib.request.urlretrieve(DEMO_URL, path)
    return path


def render_classification_page(config: dict):
    st.title("🏷️ Clasificación de Audio")
    st.markdown(
        """
        El encoder de AudioMAE extrae un **embedding global de 768 dimensiones** 
        que captura la semántica del audio.  
        Ese vector se proyecta a probabilidades de clase mediante una capa lineal.

        > ⚠️ Sin fine-tuning, la capa lineal es aleatoria (seed=42).  
        > La distribución refleja la separabilidad geométrica del espacio latente.
        """
    )

    # ── Audio ─────────────────────────────────────────────────────────────────
    st.subheader("1️⃣ Seleccionar Audio")
    audio_source = st.radio(
        "Fuente:",
        ["🎧 Demo (perro ladrando)", "📁 Subir .wav"],
        horizontal=True,
        key="cls_audio_source",
    )

    audio_path = None
    if audio_source.startswith("🎧"):
        audio_path = _get_demo_audio()
        st.success(f"Audio de demo: `{DEMO_NAME}`")
        st.audio(audio_path)
    else:
        uploaded = st.file_uploader("Archivo .wav", type=["wav"], key="cls_uploader")
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(uploaded.read())
            tmp.close()
            audio_path = tmp.name
            st.audio(audio_path)

    if audio_path is None:
        st.info("⬆️ Sube un audio para continuar.")
        return

    # ── Clases personalizables ────────────────────────────────────────────────
    st.subheader("2️⃣ Configurar Clases")
    with st.expander("Editar lista de clases (una por línea)", expanded=False):
        classes_text = st.text_area(
            "Clases objetivo:",
            value="\n".join(DEFAULT_CLASSES),
            height=200,
        )
    class_names = [c.strip() for c in classes_text.split("\n") if c.strip()]
    st.caption(f"Usando {len(class_names)} clases.")

    # ── Modelo ────────────────────────────────────────────────────────────────
    st.subheader("3️⃣ Cargar Modelo")
    model = load_model(config["checkpoint_path"])
    if model is None:
        return
    device = get_device()
    st.success(f"✅ Modelo en **{device}**")

    # ── Inferencia ────────────────────────────────────────────────────────────
    st.subheader("4️⃣ Ejecutar Clasificación")
    run_btn = st.button("▶️ Clasificar Audio", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Preprocesando audio…"):
            fbank = load_and_convert(audio_path)
            fbank_norm = normalize(fbank)

        with st.spinner("Extrayendo embedding y clasificando…"):
            result = run_classification(model, fbank_norm, class_names, device)

        # ── Resultado principal ───────────────────────────────────────────────
        st.subheader("5️⃣ Resultado")

        col1, col2, col3 = st.columns(3)
        col1.metric("Clase predicha", result.top_class)
        col2.metric("Confianza", f"{result.top_prob:.1f}%")
        col3.metric("Norma del Embedding", f"{result.embedding_norm:.4f}")

        st.divider()

        # Barras de probabilidad
        st.subheader("Distribución de Probabilidades")
        fig_bars = plot_classification_bars(class_names, result.probabilities)
        st.pyplot(fig_bars)

        # ── Embedding ─────────────────────────────────────────────────────────
        st.subheader("6️⃣ Vector Latente del Encoder")
        col_emb, col_heat = st.columns([1, 1])

        with col_emb:
            st.markdown("**Estadísticas del embedding (768-dim):**")
            emb = result.embedding
            st.write({
                "Media": f"{emb.mean():.4f}",
                "Std": f"{emb.std():.4f}",
                "Min": f"{emb.min():.4f}",
                "Max": f"{emb.max():.4f}",
                "Norma L2": f"{emb.norm():.4f}",
            })
            st.caption(
                "El embedding captura la estructura temporal y frecuencial del audio. "
                "Audios similares tendrán embeddings más cercanos (distancia coseno baja)."
            )

        with col_heat:
            fig_heat = plot_embedding_heatmap(emb)
            st.pyplot(fig_heat)

        # ── Spectrogram de referencia ─────────────────────────────────────────
        with st.expander("Ver mel-spectrogram procesado", expanded=False):
            fig_spec = plot_spectrogram(fbank_norm, title="Mel-Spectrogram (entrada al encoder)")
            st.pyplot(fig_spec)

        # ── Nota sobre el proceso ─────────────────────────────────────────────
        with st.expander("📖 ¿Cómo funciona la clasificación con AudioMAE?"):
            st.markdown(
                """
                ### Proceso de Clasificación

                1. **Preprocesado**: el audio `.wav` → mel-spectrogram (128 bandas, 1024 frames).
                2. **Encoder (ViT-Base)**: el spectrogram se divide en **512 parches de 16×16**.
                   El encoder procesa *todos* los parches (mask_ratio=0) y genera un tensor  
                   de forma `(1, 512, 768)` — 512 embeddings de 768 dimensiones.
                3. **Mean Pooling**: se promedia sobre los 512 parches → vector global `(1, 768)`.
                4. **Normalización L2**: el vector se normaliza a la hiperesfera unitaria.
                5. **Proyección lineal**: una capa `Linear(768, N_clases)` mapea el embedding  
                   al espacio de clases → **logits**.
                6. **Softmax**: convierte los logits en probabilidades.

                ### Tensores Q, K, V en la Clasificación
                En cada capa del encoder, la auto-atención genera:
                - **Q (Queries)**: representación de cada parche buscando información relevante.
                - **K (Keys)**: lo que cada parche ofrece a las queries de otros parches.
                - **V (Values)**: el contenido que se propaga cuando K es relevante para Q.

                La atención `softmax(QKᵀ/√d)·V` permite que cada parche agregue contexto
                de todos los demás, capturando relaciones globales en el audio.
                """
            )
