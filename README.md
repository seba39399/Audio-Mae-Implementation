# AudioMAE — Implementación de Inferencia con Interfaz Interactiva

**Procesamiento de Datos Secuenciales - Proyecto Final**

Integrantes: Valentina Lopez Maldonado - Yerson David Rozo - Juan Sebastián Peña

Implementación del modelo **AudioMAE (Masked Autoencoders that Listen, NeurIPS 2022)** para
inferencia, con una interfaz gráfica en Streamlit que permite cargar audio y observar el
funcionamiento del modelo en dos tareas: reconstrucción de espectrogramas y clasificación de sonidos.

---

## 1. Resumen (Abstract)

Este trabajo aplica la arquitectura Transformer encoder–decoder **AudioMAE** a una tarea de
procesamiento de datos secuenciales de audio. AudioMAE es un autoencoder enmascarado (Masked
Autoencoder) que aprende representaciones de audio de forma auto-supervisada: convierte una grabación
en un mel-espectrograma, lo divide en parches, oculta una gran fracción de ellos (típicamente 75–80 %)
y entrena un encoder–decoder Transformer para reconstruir las partes ocultas.

No se entrena el modelo desde cero. Se **reutilizan los pesos pre-entrenados** publicados por los
autores (encoder ViT-Base entrenado sobre AudioSet-2M, unos 2 millones de clips de YouTube) y se
implementa únicamente el **proceso de inferencia**. La solución incluye una interfaz en Streamlit con
tres secciones: reconstrucción de audio (tarea generativa), clasificación a partir de los embeddings
del encoder, y una explicación interactiva de la arquitectura.

Como resultado, el modelo es capaz de reconstruir mel-espectrogramas a partir de entradas con un alto
porcentaje de información eliminada, recuperando estructuras como armónicos y formantes. La interfaz
reporta métricas cuantitativas de la reconstrucción (MSE y SNR) y permite visualizar lado a lado el
espectrograma original, el enmascarado y el reconstruido.

---

## 2. Introducción

### Artículo base

- **Título:** *Masked Autoencoders that Listen*
- **Autores:** Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba,
  Florian Metze, Christoph Feichtenhofer (Meta AI, Carnegie Mellon University)
- **Publicación:** NeurIPS 2022
- **Enlace al artículo:** https://arxiv.org/abs/2207.06405
- **Repositorio original:** https://github.com/facebookresearch/AudioMAE

### Contexto del problema

El audio es un tipo de dato secuencial complejo. Tradicionalmente, los modelos de audio se inicializaban
con pesos pre-entrenados en imágenes (ImageNet), lo cual es subóptimo porque un espectrograma y una
imagen natural tienen propiedades muy distintas. Además, los Transformers tienen un costo computacional
**cuadrático** respecto a la longitud de la secuencia, lo que dificulta entrenar con secuencias largas
de audio.

### Motivación

AudioMAE resuelve ambos problemas. Por un lado, hace **pre-entrenamiento auto-supervisado solo con
audio** (sin etiquetas ni datos de otra modalidad). Por otro, al enmascarar y descartar la mayoría de
los parches, el encoder procesa solo una pequeña fracción de la secuencia, reduciendo drásticamente el
cómputo. Con esto, el modelo alcanza el estado del arte en seis tareas de clasificación de audio y voz.

### Objetivo

Comprender en profundidad la arquitectura Transformer encoder–decoder de AudioMAE, implementar su
proceso de inferencia usando los pesos pre-entrenados, y construir una herramienta de visualización
interactiva que permita cargar audio propio y observar el funcionamiento del modelo.

---

## 3. Marco teórico

### 3.1 La arquitectura general

AudioMAE es una extensión al dominio del audio del modelo MAE de imágenes. Su flujo es:

```
Audio (.wav)
   |
Mel-Espectrograma  (1024 frames de tiempo x 128 bandas mel)
   |
División en parches de 16x16  ->  512 parches en total
   |
Enmascaramiento aleatorio (por ejemplo, se oculta el 75%)
   |
ENCODER (ViT-Base, 12 capas, dim 768, 12 cabezas)
   Procesa SOLO los parches visibles (~25%)
   |
   Latentes de los parches visibles
   |
DECODER (Transformer, 16 capas, dim 512, atención local con ventanas desplazadas)
   Recibe los latentes + "mask tokens" en las posiciones ocultas
   Reordena la secuencia y reconstruye cada parche oculto
   |
Mel-Espectrograma reconstruido (1024 x 128)
```

La entrada y la salida del modelo son mel-espectrogramas. El objetivo de entrenamiento es minimizar el
**error cuadrático medio (MSE)** entre los parches reconstruidos y los originales, calculado únicamente
sobre los parches que fueron ocultados.

### 3.2 El mecanismo de atención

El núcleo del Transformer es la **auto-atención (self-attention)**. Permite que cada parche del
espectrograma "mire" a todos los demás parches y decida de cuáles tomar más información. La fórmula es:

```
Attention(Q, K, V) = softmax( (Q · K^T) / sqrt(d_k) ) · V
```

donde `d_k = 64` (768 dimensiones repartidas entre 12 cabezas). El factor `sqrt(d_k)` en el
denominador evita que los productos crezcan demasiado y saturen el softmax (lo que dejaría gradientes
casi nulos durante el entrenamiento).

El modelo usa **atención multi-cabeza (Multi-Head Attention)**: en lugar de una sola atención, ejecuta
12 atenciones en paralelo y concatena sus resultados. Cada cabeza puede especializarse en distintos
patrones, por ejemplo relaciones de baja frecuencia, transiciones de inicio/fin de sonido, o armónicos
periódicos.

### 3.3 Generación de los tensores Q, K y V

Dado un tensor de entrada `X` con forma `(N, 768)` (N parches, 768 dimensiones cada uno), cada parche
genera tres vectores mediante proyecciones lineales aprendidas durante el entrenamiento:

```
Q = X · W_Q    Query  -> "qué estoy buscando"
K = X · W_K    Key    -> "cómo me identifico para ser encontrado"
V = X · W_V    Value  -> "qué información aporto"
```

- **Q (Query):** lo que un parche necesita de los demás para construir su salida.
- **K (Key):** la "etiqueta" con la que cada parche se anuncia frente a los demás.
- **V (Value):** el contenido real que un parche entrega cuando es atendido.

El producto `Q · K^T` mide la afinidad entre cada par de parches; el softmax la convierte en pesos que
suman 1; y finalmente esos pesos se aplican sobre `V` para obtener la salida ponderada. En la práctica,
las tres proyecciones se calculan juntas con una sola capa lineal `nn.Linear(dim, dim*3)` y luego se
separan en Q, K y V.

### 3.4 Innovaciones del modelo

1. **Ratio de enmascaramiento muy alto (75–80 %).** A diferencia de BERT en texto (15 %), el audio
   tiene mucha redundancia, por lo que ocultar la mayoría obliga al modelo a aprender representaciones
   ricas en lugar de copiar de vecinos cercanos.
2. **Encoder asimétrico y eficiente.** El encoder solo procesa los parches visibles, no los ocultos,
   reduciendo el cómputo alrededor de un 75 %. Esto permite usar un encoder grande (ViT-Base) a un
   costo razonable.
3. **Decoder profundo con atención local (shifted window).** A diferencia del MAE de imágenes (que
   usa un decoder pequeño de 8 capas con atención global), Audio-MAE emplea un decoder de **16 capas**
   con atención por ventanas locales desplazadas. La razón: en un espectrograma la posición en tiempo
   y frecuencia sí importa (formantes y armónicos están agrupados localmente), por lo que la atención
   local es más adecuada que la global. Como cada capa atiende solo a una ventana pequeña (4×4 parches
   en lugar de los 64×8 globales), se pueden apilar más capas con poco costo extra, lo que mejora la
   reconstrucción. El paper muestra que con atención global lo óptimo serían 8 capas, pero con atención
   local lo es 16.
4. **Pre-entrenamiento solo con audio.** No depende de datos de imagen (ImageNet); los autores muestran
   que el pre-entrenamiento dentro del mismo dominio (audio) da mejores resultados que transferir desde
   imágenes.
5. **Objetivo de reconstrucción simple.** Solo se usa la pérdida MSE; añadir objetivos contrastivos no
   mejora los resultados.

---

## 4. Metodología

### Herramientas utilizadas

- **Python 3.10**
- **PyTorch** y **torchaudio** — definición del modelo e inferencia.
- **timm 0.4.12** — bloques de Vision Transformer requeridos por el repositorio original.
- **soundfile** y **librosa** — lectura y manejo de audio.
- **Streamlit** — interfaz gráfica interactiva.
- **matplotlib** y **numpy** — visualización y cálculo de métricas.

### Uso de pesos pre-entrenados

El proyecto no entrena ningún modelo. Se carga el repositorio oficial de AudioMAE como código base y se
le aplican parches automáticos de compatibilidad (en [core/model_loader.py](core/model_loader.py))
para que funcione con versiones modernas de las librerías y en Windows:

- Se desactiva un import de `SwinTransformerBlock` no disponible en la versión de timm usada.
- Se reemplaza `torch._six.inf` por `math.inf`.
- Se corrige `np.float` (eliminado en NumPy moderno).
- Se ajusta una llamada `.cuda()` fija para que use el dispositivo disponible (CPU o GPU).
- Se adapta `PosixPath` a `WindowsPath` para cargar el checkpoint.

Los pesos (encoder ViT-Base pre-entrenado sobre AudioSet-2M) se descargan desde un enlace de Google
Drive y se cargan con `torch.load`. La carga se hace una sola vez gracias a `st.cache_resource`.

### Organización del código

```
Audio-Mae-Implementation/
├── app.py                      # Punto de entrada de Streamlit
├── requirements.txt
├── core/
│   ├── preprocessing.py        # Audio .wav -> mel-espectrograma (Kaldi FBANK)
│   ├── model_loader.py         # Carga, parcheo y cacheo del modelo pre-entrenado
│   ├── reconstruction.py       # Inferencia de reconstrucción + métricas
│   └── classification.py       # Extracción de embeddings y clasificación
├── ui/
│   ├── sidebar.py              # Navegación y configuración global
│   ├── page_reconstruction.py  # Página de reconstrucción
│   ├── page_classification.py  # Página de clasificación
│   └── page_architecture.py    # Explicación teórica interactiva
├── utils/
│   └── visualization.py        # Gráficas de espectrogramas y métricas
├── notebook/
│   └── AudioMAE_Reconstruction_Demo.ipynb
└── AudioMAE/                   # Repositorio oficial clonado (código base + pesos)
```

---

## 5. Desarrollo e implementación

### Pasos para ejecutar el proyecto

```bash
# 1. Clonar el repositorio oficial de AudioMAE dentro de la carpeta del proyecto
git clone https://github.com/facebookresearch/AudioMAE.git

# 2. Instalar las dependencias
pip install -r requirements.txt

# 3. Descargar los pesos pre-entrenados (ViT-B, AudioSet-2M, ~330 MB) desde:
#    https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu
#    y guardarlos en:  AudioMAE/ckpt/pretrained.pth

# 4. Lanzar la aplicación
streamlit run app.py
```

### Cómo se cargan los pesos

La función `load_model` en [core/model_loader.py](core/model_loader.py):

1. Localiza el repositorio `AudioMAE/` y lo agrega al path de Python.
2. Aplica los parches de compatibilidad descritos en la metodología.
3. Construye la arquitectura con `models_mae.mae_vit_base_patch16(in_chans=1, audio_exp=True, img_size=(1024,128))`.
4. Carga el archivo `.pth` con `torch.load(..., map_location="cpu")`, extrae el diccionario de pesos y
   lo asigna con `load_state_dict(strict=False)`.
5. Mueve el modelo a GPU si está disponible y lo pone en modo `eval()`.

### Preprocesamiento

En [core/preprocessing.py](core/preprocessing.py), siguiendo los parámetros exactos del artículo:

1. Se lee el `.wav` y se convierte a mono.
2. Se elimina el offset de continua (normalización DC).
3. Si el audio dura menos de ~10 s, se repite cíclicamente (evita el padding con ceros, que degrada la
   reconstrucción).
4. Se calcula el banco de filtros mel **Kaldi FBANK** con 128 bandas, ventana Hanning de 25 ms y salto
   de 10 ms.
5. Se recorta o rellena a exactamente **1024 frames × 128 bandas**.
6. Se normaliza con la media (−4.268) y desviación estándar (4.569) del artículo.

El resultado es el tensor de entrada del modelo con forma `(1, 1, 1024, 128)` = `(batch, canal, tiempo, frecuencia)`.

### Proceso de inferencia

**Reconstrucción** ([core/reconstruction.py](core/reconstruction.py)):

1. Se pasa el espectrograma normalizado por `model(x, mask_ratio)`, que internamente parchea, enmascara,
   codifica con el encoder y reconstruye con el decoder.
2. Con `model.unpatchify(...)` se reconstruye el espectrograma 2D a partir de los parches predichos.
3. Se expande la máscara binaria a la resolución del espectrograma para poder visualizar qué se ocultó.
4. Se calculan las métricas: pérdida MSE en parches ocultos, MSE global, SNR (relación señal/ruido en dB)
   y el porcentaje real enmascarado.

**Clasificación** ([core/classification.py](core/classification.py)):

1. Se pasa el espectrograma por `model.forward_encoder(x, mask_ratio=0.0)` (sin ocultar nada, el
   encoder ve todos los parches).
2. Se hace *mean pooling* sobre los parches (sin el token CLS) para obtener un embedding global de 768
   dimensiones, que se normaliza con norma L2.
3. Ese embedding pasa por una capa lineal para producir logits y probabilidades por clase. Como no se
   hizo fine-tuning, esta capa es aleatoria con semilla fija (42); sirve para demostrar la geometría del
   espacio latente, no como clasificador final entrenado.

---

## 6. Resultados y análisis

### Reconstrucción

La página de reconstrucción muestra cuatro paneles lado a lado:

| Panel | Qué muestra |
|-------|-------------|
| (a) Original | Mel-espectrograma real del audio de entrada |
| (b) Enmascarado | Lo que ve el encoder: los parches ocultos aparecen en zona neutra |
| (c) Combinación | Parches visibles originales + parches reconstruidos por el decoder |
| (d) Solo reconstruido | Salida pura del decoder |

La interfaz reporta las siguientes métricas para cada inferencia:

- **Loss (MSE en parches ocultos):** error de reconstrucción solo en lo que el modelo no vio.
- **MSE Global:** error promedio sobre todo el espectrograma.
- **SNR (dB):** relación señal/ruido de la reconstrucción; valores más altos indican mejor calidad.
- **% real enmascarado:** porcentaje efectivo de parches ocultados.

**Análisis esperado:** al aumentar el `mask_ratio`, la tarea se vuelve más difícil, el MSE tiende a subir
y el SNR a bajar. Aun con 75–80 % de parches ocultos, el modelo recupera estructuras globales como
armónicos (líneas horizontales en frecuencia) y la envolvente temporal del sonido. Los sonidos con
patrones repetitivos (música, eventos) se reconstruyen mejor que la voz, que es más impredecible —
coincidiendo con lo reportado en el artículo.

```
[ Pendiente imagen: paneles de reconstrucción + tabla de métricas ]
```

### Clasificación

La página de clasificación muestra el embedding del encoder, las probabilidades por clase y la clase
con mayor puntaje, junto con la norma L2 del embedding.

**Análisis esperado:** como la cabeza de clasificación no está entrenada (capa lineal aleatoria), las
predicciones no son fiables como etiquetas reales. El valor de esta sección es **demostrativo**: muestra
que el encoder produce un embedding global estable y de dimensión 768 a partir de cualquier audio, que
es justo lo que se usaría como punto de partida para un fine-tuning supervisado.

```
[ Pendiente imagen de embedding + barras de probabilidad por clase ]
```

---

## 7. Conclusiones

### Aprendizajes

- Se comprendió cómo un mismo principio (enmascarar y reconstruir) aplicado en BERT (texto) y MAE
  (imágenes) se extiende al audio mediante espectrogramas.
- Se entendió en detalle el flujo de un Transformer encoder–decoder: parcheo, generación de Q/K/V,
  atención escalada multi-cabeza, y reconstrucción a partir de tokens de máscara.
- Se vio el valor de la **asimetría** del modelo: un encoder que procesa solo lo visible permite usar
  Transformers grandes con un costo manejable.

### Limitaciones

- **Resolución fija:** el modelo espera exactamente 1024 × 128; audios de otra duración se recortan o
  repiten, lo que introduce artefactos.
- **La salida es un espectrograma, no audio:** para escuchar el resultado se necesitaría un vocoder
  (por ejemplo, Griffin-Lim o HiFi-GAN).
- **Clasificación sin entrenar:** la capa lineal es aleatoria, por lo que no refleja la capacidad real
  del encoder sin fine-tuning.
- **Costo computacional:** la inferencia es cómoda con GPU; en CPU es más lenta, y el checkpoint pesa
  ~330 MB.

### Posibles mejoras

- Integrar un vocoder para reproducir el audio reconstruido.
- Hacer fine-tuning del encoder con un dataset etiquetado (por ejemplo ESC-50) para obtener clasificación
  real.
- Permitir audios de duración variable mediante ventanas deslizantes.
- Visualizar los mapas de atención de las cabezas del encoder para análisis interpretativo.

---

## 8. Referencias

[1] P.-Y. Huang, H. Xu, J. Li, A. Baevski, M. Auli, W. Galuba, F. Metze, and C. Feichtenhofer,
"Masked Autoencoders that Listen," in *Proc. 36th Conf. Neural Information Processing Systems (NeurIPS)*,
2022. [En línea]. Disponible: https://arxiv.org/abs/2207.06405

[2] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, "Masked Autoencoders Are Scalable Vision
Learners," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin,
"Attention Is All You Need," in *Proc. 31st Int. Conf. Neural Information Processing Systems (NIPS)*,
2017, pp. 6000–6010.

[4] A. Dosovitskiy *et al.*, "An Image Is Worth 16x16 Words: Transformers for Image Recognition at
Scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.

[6] Y. Gong, Y.-A. Chung, and J. Glass, "AST: Audio Spectrogram Transformer," in *Proc. Interspeech*,
2021, pp. 571–575.

[7] J. F. Gemmeke *et al.*, "Audio Set: An Ontology and Human-Labeled Dataset for Audio Events," in
*Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2017, pp. 776–780.

[8] Repositorio oficial AudioMAE, Facebook Research. [En línea]. Disponible:
https://github.com/facebookresearch/AudioMAE
