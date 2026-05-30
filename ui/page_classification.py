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
from utils.visualization import plot_embedding_heatmap, plot_spectrogram

DEMO_URL  = "https://github.com/karolpiczak/ESC-50/raw/master/audio/1-100038-A-14.wav"
DEMO_NAME = "sample_dog_bark.wav"


def _get_demo_audio() -> str:
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, DEMO_NAME)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        with st.spinner("Descargando audio de demo…"):
            urllib.request.urlretrieve(DEMO_URL, path)
    return path


def _plot_topk_bars(top_k: list):
    """Gráfico de barras horizontal para el Top-K."""
    import matplotlib.pyplot as plt

    names = [d["class"] for d in top_k]
    probs = [d["prob"] for d in top_k]
    colors = ["#e74c3c"] + ["#3498db"] * (len(names) - 1)

    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.55)))
    bars = ax.barh(names[::-1], probs[::-1], color=colors[::-1], height=0.6)
    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confianza / Score (%)")
    ax.set_title("Top predicciones", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def render_classification_page(config: dict):
    st.title("🏷️ Clasificación de Audio")

    # ── Modo de clasificación ─────────────────────────────────────────────────
    ft_ckpt = os.path.join(os.path.dirname(config["checkpoint_path"]), "finetuned.pth")
    ft_available = os.path.exists(ft_ckpt)

    if ft_available:
        st.success(
            "✅ Checkpoint fine-tuned detectado (`ckpt/finetuned.pth`)  \n"
            "Usando clasificación real con **527 clases de AudioSet**."
        )
        mode_label = "🎯 Fine-tuned (AudioSet 527 clases) — REAL"
    else:
        st.warning(
            "⚠️ No se encontró `AudioMAE/ckpt/finetuned.pth`.  \n"
            "Usando modo prototipo (similitud coseno). Para clasificación real, "
            "descarga el checkpoint fine-tuned desde "
            "[Google Drive](https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view) "
            "y guárdalo en `AudioMAE/ckpt/finetuned.pth`."
        )
        mode_label = "📐 Prototipo (similitud coseno)"

    st.caption(f"Modo activo: **{mode_label}**")
    st.markdown(
        """
        El encoder de AudioMAE extrae un **embedding global de 768 dimensiones**.  
        En modo fine-tuned, una cabeza lineal real (entrenada en AudioSet) convierte
        ese embedding en scores por clase usando **sigmoid** (clasificación multi-label).
        """
    )

    # ── Audio ─────────────────────────────────────────────────────────────────
    st.subheader("1️⃣ Seleccionar Audio")
    audio_source = st.radio(
        "Fuente:",
        ["🎧 Demo (perro ladrando — ESC-50)", "📁 Subir mi propio .wav"],
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

    # ── Clases (solo modo prototipo) ──────────────────────────────────────────
    if not ft_available:
        st.subheader("2️⃣ Configurar Clases")
        with st.expander("Editar lista de clases (una por línea)", expanded=False):
            classes_text = st.text_area(
                "Clases objetivo:",
                value="\n".join(DEFAULT_CLASSES),
                height=200,
            )
        class_names = [c.strip() for c in classes_text.split("\n") if c.strip()]
        st.caption(f"Usando {len(class_names)} clases personalizadas.")
    else:
        class_names = None   # se usan las 527 de AudioSet

    # ── Modelo ────────────────────────────────────────────────────────────────
    st.subheader("3️⃣ Cargar Encoder")
    model = load_model(config["checkpoint_path"])
    if model is None:
        return
    device = get_device()
    st.success(f"✅ Encoder cargado — corriendo en **{device}**")

    # ── Inferencia ────────────────────────────────────────────────────────────
    st.subheader("4️⃣ Ejecutar Clasificación")
    if not ft_available:
        st.info("⏱️ Primera ejecución ~30 s (genera prototipos por clase).")

    run_btn = st.button("▶️ Clasificar Audio", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Preprocesando audio…"):
            fbank     = load_and_convert(audio_path)
            fbank_norm = normalize(fbank)

        with st.spinner("Clasificando…"):
            result = run_classification(
                model,
                fbank_norm,
                class_names=class_names,
                device=device,
                finetuned_ckpt=ft_ckpt if ft_available else None,
            )

        # ── Resultado ─────────────────────────────────────────────────────────
        st.subheader("5️⃣ Resultado")
        badge = "🎯 Fine-tuned (AudioSet)" if result.mode == "finetuned" else "📐 Prototipo"
        st.caption(f"Modo usado: **{badge}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Clase #1", result.top_class)
        col2.metric("Confianza", f"{result.top_prob:.1f}%")
        col3.metric("Norma Embedding", f"{result.embedding_norm:.4f}")

        st.divider()

        # Top-K
        st.subheader(f"Top {len(result.top_k)} predicciones")
        fig_bars = _plot_topk_bars(result.top_k)
        st.pyplot(fig_bars)

        # Tabla detallada
        with st.expander("Ver tabla completa de predicciones"):
            for i, d in enumerate(result.top_k, 1):
                st.write(f"**{i}.** {d['class']} — `{d['prob']:.2f}%`")

        # ── Embedding ─────────────────────────────────────────────────────────
        st.subheader("6️⃣ Vector Latente del Encoder (768-dim)")
        col_emb, col_heat = st.columns([1, 1])

        with col_emb:
            emb = result.embedding
            st.write({
                "Media":    f"{emb.mean():.4f}",
                "Std":      f"{emb.std():.4f}",
                "Min":      f"{emb.min():.4f}",
                "Max":      f"{emb.max():.4f}",
                "Norma L2": f"{emb.norm():.4f}",
            })
            st.caption(
                "Embedding extraído por mean-pooling sobre los 512 parches del encoder. "
                "Audios similares producen embeddings con alta similitud coseno."
            )

        with col_heat:
            fig_heat = plot_embedding_heatmap(emb)
            st.pyplot(fig_heat)

        # Spectrogram
        with st.expander("Ver mel-spectrogram procesado", expanded=False):
            fig_spec = plot_spectrogram(fbank_norm, title="Mel-Spectrogram (entrada al encoder)")
            st.pyplot(fig_spec)

        # Explicación
        with st.expander("📖 ¿Cómo funciona la clasificación fine-tuned?"):
            st.markdown(
                """
                ### Proceso con checkpoint fine-tuned

                1. **Encoder** (ViT-Base, 12 capas): procesa los 512 parches del spectrogram
                   con `mask_ratio=0.0` → tensor `(1, 513, 768)`.
                2. **Mean Pooling**: promedio sobre los 512 parches de audio (excluye CLS) → `(1, 768)`.
                3. **Cabeza lineal** `Linear(768 → 527)`: los pesos vienen del fine-tuning
                   en AudioSet-2M con etiquetas reales.
                4. **Sigmoid** (no softmax): AudioSet es multi-label → un audio puede
                   pertenecer a varias clases simultáneamente (ej: "Dog" + "Bark").
                5. **Top-K**: se muestran las 10 clases con mayor score.

                ### Q, K, V en el encoder
                - **Q**: qué información busca cada parche del spectrogram.
                - **K**: cómo se identifica cada parche para ser encontrado.
                - **V**: el contenido que aporta al resultado ponderado.
                
                La atención `softmax(QKᵀ/√64)·V` permite que cada parche
                integre contexto de todo el spectrogram antes de clasificar.
                """
            )
