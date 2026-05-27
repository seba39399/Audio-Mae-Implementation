# AudioMAE — Demo Interactivo con Streamlit

Valentina Lopez - Yerson Rozo - Juan Peña

Implementación modular del modelo **AudioMAE** (Masked Autoencoders that Listen, NeurIPS 2022)
con interfaz gráfica en Streamlit para dos tareas:

- 🔄 **Reconstrucción** (tarea de generación): el modelo reconstruye parches enmascarados del mel-spectrogram.
- 🏷️ **Clasificación**: el encoder extrae embeddings para clasificar el audio por categoría.

## Estructura del proyecto

```
audiomae_app/
├── app.py                    # Punto de entrada Streamlit
├── requirements.txt
├── core/
│   ├── preprocessing.py      # Audio → mel-spectrogram (Kaldi FBANK)
│   ├── model_loader.py       # Carga y cacheo del modelo pre-entrenado
│   ├── reconstruction.py     # Inferencia de reconstrucción
│   └── classification.py     # Extracción de embeddings y clasificación
├── ui/
│   ├── sidebar.py            # Navegación y configuración global
│   ├── page_reconstruction.py
│   ├── page_classification.py
│   └── page_architecture.py  # Explicación teórica interactiva
└── utils/
    └── visualization.py      # Plots de spectrogramas, barras, heatmaps
```

## Instalación

```bash
# 1. Clonar el repo de AudioMAE (en la misma carpeta que audiomae_app/)
git clone https://github.com/facebookresearch/AudioMAE.git

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar pesos pre-entrenados (ViT-B, AudioSet-2M ~330 MB)
# https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu
# Guardar en: AudioMAE/ckpt/pretrained.pth

# 4. Ejecutar
streamlit run app.py
```

## Referencia

He, P., et al. "Masked Autoencoders that Listen." NeurIPS 2022.  
https://arxiv.org/abs/2207.06405  
Repositorio original: https://github.com/facebookresearch/AudioMAE
