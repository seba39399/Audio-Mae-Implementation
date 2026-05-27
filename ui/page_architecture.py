"""
ui/page_architecture.py
Página explicativa de la arquitectura AudioMAE (sin necesidad del modelo cargado).
"""

import streamlit as st


def render_architecture_page():
    st.title("🧠 Arquitectura AudioMAE — Explicación Detallada")

    tabs = st.tabs([
        "📐 Visión General",
        "🔍 Mecanismo de Atención",
        "⚡ Q, K, V en detalle",
        "🚀 Innovaciones",
        "⚠️ Limitaciones",
    ])

    # ── Tab 1: Visión General ─────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(
            """
            ## AudioMAE: Masked Autoencoders that Listen
            **Paper**: He et al., NeurIPS 2022  
            **Base**: Vision Transformer (ViT) adaptado a audio via mel-spectrograms

            ### Pipeline completo

            ```
            Audio (.wav)
               ↓
            Mel-Spectrogram  (1024 × 128)
               ↓
            Parcheo (patch_size = 16×16)  →  512 parches
               ↓ 
            Enmascaramiento aleatorio (mask_ratio, ej: 75%)
               ↓
            ┌─────────────────────────────────────────────────────┐
            │              ENCODER (ViT-Base)                     │
            │  Solo procesa los 25% de parches VISIBLES           │
            │  12 capas de Transformer con Multi-Head Attention   │
            │  Embedding dim = 768, 12 heads                      │
            └──────────────────┬──────────────────────────────────┘
                               │ Latents (128 parches visibles × 768-dim)
                               ↓
            ┌─────────────────────────────────────────────────────┐
            │              DECODER (ViT-Small)                    │
            │  Recibe: latents visibles + mask tokens             │
            │  8 capas Transformer, embedding dim = 512           │
            │  Predice: valor de cada parche oculto               │
            └──────────────────┬──────────────────────────────────┘
                               │ Reconstrucción (512 parches × 256-dim)
                               ↓
            Mel-Spectrogram reconstruido  (1024 × 128)
            ```

            ### Dimensiones clave
            | Componente | Valor |
            |---|---|
            | Input spectrogram | 1024 × 128 (T × F) |
            | Patch size | 16 × 16 |
            | Número de parches | 512 |
            | Parches visibles (mask=0.75) | ~128 |
            | Parches ocultos (mask=0.75) | ~384 |
            | Embedding del encoder | 768-dim |
            | Embedding del decoder | 512-dim |
            | Capas encoder | 12 |
            | Capas decoder | 8 |
            | Cabezas de atención (encoder) | 12 |
            """
        )

    # ── Tab 2: Mecanismo de Atención ──────────────────────────────────────────
    with tabs[1]:
        st.markdown(
            """
            ## Multi-Head Self-Attention (MHSA)

            ### ¿Qué hace la atención?
            Permite que **cada parche del spectrogram "escuche" a todos los demás** parches
            y pondere cuánta información tomar de cada uno, de manera diferenciada por cabeza.

            ### Fórmula central
            """
        )
        st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V")

        st.markdown(
            """
            Donde `dₖ = 64` (768 dim / 12 heads).

            ### Por qué `√dₖ` en el denominador
            Sin el factor de escala, el producto `QKᵀ` crece en magnitud con la dimensión,
            empujando el softmax a regiones de gradiente casi nulo. Dividir por `√dₖ`
            estabiliza el entrenamiento.

            ### Multi-Head: atención paralela
            En lugar de una sola atención, el modelo usa **12 cabezas independientes**:

            ```
            MultiHead(Q, K, V) = Concat(head₁, head₂, …, head₁₂) · Wᴼ
            headᵢ = Attention(Q·Wᵢᴼ, K·Wᵢᴷ, V·Wᵢᵛ)
            ```

            Cada cabeza puede especializarse en **distintos patrones temporales o frecuenciales**:
            - Cabeza 1 → relaciones de baja frecuencia
            - Cabeza 7 → relaciones de onset/offset
            - Cabeza 12 → harmónicos periódicos

            ### Atención en el Encoder vs Decoder
            | Componente | Tipo de atención |
            |---|---|
            | Encoder | Self-attention entre parches **visibles** |
            | Decoder | Self-attention + **Cross-attention** (latents del encoder → mask tokens) |
            """
        )

    # ── Tab 3: Q, K, V ────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(
            """
            ## Generación de Q, K, V

            ### ¿De dónde vienen?
            Dado un tensor de entrada `X` de forma `(N, 768)` (N parches),
            cada token genera sus propios Q, K, V mediante proyecciones lineales aprendidas:

            ```python
            Q = X @ W_Q   # (N, d_k)  — ¿qué estoy buscando?
            K = X @ W_K   # (N, d_k)  — ¿qué puedo ofrecer?
            V = X @ W_V   # (N, d_v)  — ¿qué información entrego?
            ```

            ### Intuición semántica
            - **Q (Query)**: representación de *lo que un parche necesita* para construir su output.
            - **K (Key)**: representación de *cómo se identifica* un parche para ser encontrado.
            - **V (Value)**: el *contenido real* que un parche aporta al resultado.

            ### En código (simplificado de models_mae.py)
            ```python
            class Attention(nn.Module):
                def __init__(self, dim, num_heads=12):
                    self.num_heads = num_heads
                    head_dim = dim // num_heads          # 64
                    self.scale = head_dim ** -0.5        # 1/√64

                    # Proyección conjunta Q+K+V (3 × dim)
                    self.qkv = nn.Linear(dim, dim * 3, bias=False)
                    self.proj = nn.Linear(dim, dim)      # proyección de salida

                def forward(self, x):
                    B, N, C = x.shape
                    # Proyectar y separar en Q, K, V
                    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]   # cada uno: (B, heads, N, head_dim)

                    # Atención escalada
                    attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
                    attn = attn.softmax(dim=-1)

                    # Agregación ponderada de valores
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    return self.proj(x)
            ```

            ### Visualización conceptual de la matriz de atención
            Para N=4 parches y 1 cabeza:
            ```
            Parche │  0    1    2    3   ← Keys
            ───────┼─────────────────────
               0 Q │ 0.1  0.6  0.2  0.1
               1 Q │ 0.4  0.1  0.4  0.1
               2 Q │ 0.2  0.3  0.1  0.4
               3 Q │ 0.1  0.1  0.6  0.2
            ```
            Cada fila suma 1. El parche 0 presta mucha atención al parche 1.
            """
        )

    # ── Tab 4: Innovaciones ───────────────────────────────────────────────────
    with tabs[3]:
        st.markdown(
            """
            ## Innovaciones de AudioMAE

            ### 1. Máscara de alta ratio (75–80%)
            A diferencia de BERT (15%), AudioMAE enmascara el **75% del spectrogram**.
            El audio tiene alta redundancia temporal → el modelo debe aprender representaciones
            más ricas para reconstruir desde poca información.

            ### 2. Enmascaramiento no aleatorio (Unstructured)
            En lugar de enmascarar bloques contiguos, AudioMAE enmascara parches
            **individualmente al azar**, lo que evita que el modelo use patrones
            locales simples de interpolación.

            ### 3. Encoder asimétrico
            El encoder procesa **solo los parches visibles** (no los masks),
            reduciendo el cómputo en un 75% respecto a procesar el spectrogram completo.
            Esto permite usar un encoder ViT-Base grande con coste computacional moderado.

            ### 4. Decoder liviano
            El decoder es deliberadamente más pequeño (ViT-Small, 8 capas vs 12 del encoder),
            ya que durante fine-tuning solo se usa el **encoder** como extractor de features.

            ### 5. Pre-training en AudioSet-2M
            Entrenado en **2 millones de clips de audio** de YouTube, el modelo aprende
            representaciones de audio generalistas sin etiquetas — aprendizaje auto-supervisado puro.

            ### 6. Transferibilidad
            Un solo checkpoint pre-entrenado funciona para:
            - Clasificación de sonidos (ESC-50, AudioSet)
            - Detección de eventos de audio
            - Separación de fuentes
            - ASR (reconocimiento de voz)
            """
        )

    # ── Tab 5: Limitaciones ───────────────────────────────────────────────────
    with tabs[4]:
        st.markdown(
            """
            ## Limitaciones de AudioMAE

            ### Computacionales
            - **Requiere GPU con ≥6 GB VRAM** para inferencia fluida en audios de 10 s.
            - El checkpoint pesa ~330 MB (ViT-Base).
            - El preprocesado (FBANK Kaldi) no está optimizado para tiempo real.

            ### Arquitectónicas
            - **Resolución fija**: el modelo fue entrenado en exactamente 1024 frames × 128 mel bands.
              Audios de distinta duración deben ser recortados o repetidos, lo que introduce artefactos.
            - **Sin causalidad temporal**: la atención es bidireccional (ve futuro y pasado),
              inadecuado para síntesis en streaming.
            - **No genera audio directamente**: la salida es un mel-spectrogram, no forma de onda.
              Requiere un vocoder (ej: HiFi-GAN) para audio reproducible.

            ### De evaluación
            - La clasificación sin fine-tuning depende de una capa lineal aleatoria →
              los resultados no reflejan la capacidad real del encoder.
            - Fine-tuning requiere datasets etiquetados y entrenamiento adicional.

            ### Del paper
            - Comparado con modelos de lenguaje, el audio tiene **menor diversidad semántica**,
              lo que puede limitar el beneficio del pre-training a escala.
            - El modelo no modela **relaciones multimodales** (audio + video + texto) de forma nativa.
            """
        )
