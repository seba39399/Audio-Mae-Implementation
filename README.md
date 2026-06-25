# AudioMAE — Inference Implementation with an Interactive Interface

Valentina Lopez Maldonado - Yerson David Rozo - Juan Sebastián Peña

Implementation of the **AudioMAE (Masked Autoencoders that Listen, NeurIPS 2022)** model for
inference, with a graphical interface in Streamlit that allows users to upload audio and observe the
model’s performance on two tasks: spectrogram reconstruction and sound classification.

---

## 1. Abstract

This work applies the **AudioMAE** Transformer encoder–decoder architecture to a task involving
the processing of sequential audio data. AudioMAE is a masked autoencoder (Masked
Autoencoder) that learns audio representations in a self-supervised manner: it converts a recording
into a mel-spectrogram, divides it into patches, masks a large fraction of them (typically 75–80%)
, and trains a Transformer encoder–decoder to reconstruct the masked parts.

The model is not trained from scratch. **Pre-trained weights** published by the
authors (a ViT-Base encoder trained on AudioSet-2M, approximately 2 million YouTube clips) are reused, and
only the **inference process** is implemented. The solution includes a Streamlit interface with
three sections: audio reconstruction (generative task), classification based on the encoder’s
embeddings, and an interactive explanation of the architecture.

As a result, the model is capable of reconstructing mel-spectrograms from inputs with a high
percentage of information removed, recovering structures such as harmonics and formants. The interface
reports quantitative metrics for the reconstruction (MSE and SNR) and allows for side-by-side visualization of the
original, masked, and reconstructed spectrograms.

---

## 2. Introduction

### Original Paper

- **Title:** *Masked Autoencoders that Listen*
- **Authors:** Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba,
  Florian Metze, Christoph Feichtenhofer (Meta AI, Carnegie Mellon University)
- **Publication:** NeurIPS 2022
- **Link to the paper:** https://arxiv.org/abs/2207.06405
- **Original repository:** https://github.com/facebookresearch/AudioMAE

### Problem Context

Audio is a complex type of sequential data. Traditionally, audio models were initialized
with weights pre-trained on images (ImageNet), which is suboptimal because a spectrogram and a
natural image have very different properties. Furthermore, Transformers have a computational cost
that is **quadratic** with respect to the sequence length, making it difficult to train on long
audio sequences.

### Motivation

AudioMAE solves both problems. On the one hand, it performs **self-supervised pre-training using only
audio** (without labels or data from other modalities). On the other hand, by masking and discarding most of
the patches, the encoder processes only a small fraction of the sequence, drastically reducing the
computational cost. As a result, the model achieves state-of-the-art performance on six audio and speech classification tasks.

### Objective

To gain an in-depth understanding of AudioMAE’s Transformer encoder–decoder architecture, implement its
inference process using the pre-trained weights, and build an interactive visualization tool
that allows users to upload their own audio and observe how the model works.

---

## 3. Theoretical Framework

### 3.1 The General Architecture

AudioMAE is an extension of the image-based MAE model to the audio domain. Its workflow is as follows:

```
Audio (.wav)
   |
Mel-Spectrogram  (1024 time frames × 128 mel bands)
   |
Division into 16×16 patches  ->  512 patches in total
   |
Random masking (for example, 75% is hidden)
   |
ENCODER (ViT-Base, 12 layers, 768-dimensional, 12 heads)
   Processes ONLY the visible patches (~25%)
   |
   Latent representations of the visible patches
   |
DECODER (Transformer, 16 layers, dim 512, local attention with shifted windows)
   Receives the latent representations + “mask tokens” at the hidden positions
   Reorders the sequence and reconstructs each hidden patch
   |
Reconstructed Mel-Spectrogram (1024 x 128)
```

The model’s input and output are mel-spectrograms. The training objective is to minimize the
**mean squared error (MSE)** between the reconstructed patches and the originals, calculated solely
on the patches that were hidden.

### 3.2 The Attention Mechanism

The core of the Transformer is **self-attention**. It allows each patch of the
spectrogram to “look” at all the other patches and decide which ones to gather more information from. The formula is:

```
Attention(Q, K, V) = softmax( (Q · K^T) / sqrt(d_k) ) · V
```

where `d_k = 64` (768 dimensions distributed across 12 heads). The factor `sqrt(d_k)` in the
denominator prevents the products from growing too large and saturating the softmax (which would result in
nearly zero gradients during training).

The model uses **multi-head attention**: instead of a single attention mechanism, it runs
12 attention mechanisms in parallel and concatenates their results. Each head can specialize in different
patterns, such as low-frequency relationships, sound start/end transitions, or periodic
harmonics.

### 3.3 Generation of the Q, K, and V Tensors

Given an input tensor `X` of shape `(N, 768)` (N patches, 768 dimensions each), each patch
generates three vectors using linear projections learned during training:

```
Q = X · W_Q    Query  -> “what am I looking for”
K = X · W_K    Key    -> “how do I identify myself so I can be found”
V = X · W_V    Value  -> “what information do I provide”
```

- **Q (Query):** what a patch needs from the others to construct its output.
- **K (Key):** the “label” each patch uses to identify itself to the others.
- **V (Value):** the actual content a patch provides when it is selected.

The product `Q · K^T` measures the affinity between each pair of patches; the softmax function converts this into weights that
sum to 1; and finally, those weights are applied to `V` to obtain the weighted output. In practice,
the three projections are calculated together using a single linear layer `nn.Linear(dim, dim*3)` and then
separated into Q, K, and V.

### 3.4 Model Innovations

1. **Very high masking ratio (75–80%).** Unlike BERT for text (15%), audio
   has a lot of redundancy, so masking most of it forces the model to learn rich representations
   rather than copying from nearby neighbors.
2. **Efficient, asymmetric encoder.** The encoder processes only the visible patches, not the hidden ones,
   reducing computational cost by about 75%. This allows for the use of a large encoder (ViT-Base) at a
   reasonable cost.
3. **Deep decoder with local attention (shifted window).** Unlike image MAE (which
   uses a small 8-layer decoder with global attention), Audio-MAE employs a **16-layer** decoder
   with shifted local window attention. The reason: in a spectrogram, position in time
   and frequency does matter (formants and harmonics are clustered locally), so local attention
   is more appropriate than global attention. Since each layer attends only to a small window (4×4 patches
   instead of the global 64×8), more layers can be stacked at little extra cost, which improves
   reconstruction. The paper shows that with global attention, 8 layers would be optimal, but with
   local attention, 16 layers are optimal.
4. **Pre-training with audio only.** It does not rely on image data (ImageNet); the authors show
   that pre-training within the same domain (audio) yields better results than transfer learning from
   images.
5. **Simple reconstruction objective.** Only the MSE loss is used; adding contrastive objectives does not
   improve the results.

---

## 4. Methodology

### Tools used

- **Python 3.10**
- **PyTorch** and **torchaudio** — model definition and inference.
- **timm 0.4.12** — Vision Transformer blocks required by the original repository.
- **soundfile** and **librosa** — audio reading and handling.
- **Streamlit** — interactive graphical interface.
- **matplotlib** and **numpy** — visualization and metric calculation.

### Using Pre-trained Weights

This project does not train any models. The official AudioMAE repository is loaded as the codebase, and
automatic compatibility patches are applied (in [core/model_loader.py](core/model_loader.py))
to make it work with modern versions of the libraries and on Windows:

- An import of `SwinTransformerBlock`—which is not available in the version of timm used—is disabled.
- `torch._six.inf` is replaced with `math.inf`.
- `np.float` (removed in modern NumPy) is corrected.
- A fixed `.cuda()` call is adjusted to use the available device (CPU or GPU).
- `PosixPath` is adapted to `WindowsPath` to load the checkpoint.

The weights (a ViT-Base encoder pre-trained on AudioSet-2M) are downloaded from a Google
Drive link and loaded using `torch.load`. Loading is performed only once thanks to `st.cache_resource`.

### Code Organization

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

## 5. Development and Implementation

### Steps to run the project

```bash
# 1. Clone the official AudioMAE repository into the project folder
git clone https://github.com/facebookresearch/AudioMAE.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the pre-trained weights (ViT-B, AudioSet-2M, ~330 MB) from:
#    https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu
#    and save them to:  AudioMAE/ckpt/pretrained.pth

# 4. Run the application
streamlit run app.py
```

### How the weights are loaded

The `load_model` function in [core/model_loader.py](core/model_loader.py):

1. Locates the `AudioMAE/` repository and adds it to the Python path.
2. Applies the compatibility patches described in the methodology.
3. Constructs the architecture using `models_mae.mae_vit_base_patch16(in_chans=1, audio_exp=True, img_size=(1024,128))`.
4. Load the `.pth` file with `torch.load(..., map_location="cpu")`, extract the weight dictionary, and
   load it with `load_state_dict(strict=False)`.
5. Move the model to the GPU if available and set it to `eval()` mode.

### Preprocessing

In [core/preprocessing.py](core/preprocessing.py), following the exact parameters from the paper:

1. Read the `.wav` file and convert it to mono.
2. Remove the DC offset (DC normalization).
3. If the audio is shorter than ~10 s, it is looped cyclically (to avoid padding with zeros, which degrades
   reconstruction).
4. The **Kaldi FBANK** mel filter bank is computed with 128 bands, a 25-ms Hanning window, and a
   10-ms step size.
5. It is cropped or padded to exactly **1024 frames × 128 bands**.
6. It is normalized using the mean (−4.268) and standard deviation (4.569) from the paper.

The result is the model’s input tensor with shape `(1, 1, 1024, 128)` = `(batch, channel, time, frequency)`.

### Inference Process

**Reconstruction** ([core/reconstruction.py](core/reconstruction.py)):

1. The normalized spectrogram is passed through `model(x, mask_ratio)`, which internally patches, masks,
   encodes with the encoder, and reconstructs with the decoder.
2. Using `model.unpatchify(...)`, the 2D spectrogram is reconstructed from the predicted patches.
3. The binary mask is expanded to the spectrogram’s resolution to visualize what was hidden.
4. The metrics are calculated: MSE loss on hidden patches, global MSE, SNR (signal-to-noise ratio in dB),
   and the actual percentage masked.

**Classification** ([core/classification.py](core/classification.py)):

1. The spectrogram is passed through `model.forward_encoder(x, mask_ratio=0.0)` (without masking anything; the
   encoder sees all patches).
2. *Mean pooling* is performed on the patches (excluding the CLS token) to obtain a global embedding of 768
   dimensions, which is normalized using L2 norm.
3. That embedding passes through a linear layer to produce logits and probabilities per class. Since no
   fine-tuning was performed, this layer is random with a fixed seed (42); it serves to demonstrate the geometry of the
   latent space, not as a trained final classifier.

---

## 6. Results and Analysis



### Reconstruction

The reconstruction page displays four panels side by side:

| Panel | What it shows |
|-------|-------------|
| (a) Original | Actual Mel-spectrogram of the input audio |
| (b) Masked | What the encoder sees: hidden patches appear in the neutral zone |
| (c) Combination | Original visible patches + patches reconstructed by the decoder |
| (d) Reconstructed Only | Pure output from the decoder |

The interface reports the following metrics for each inference:

- **Loss (MSE on hidden patches):** reconstruction error limited to what the model did not see.
- **Global MSE:** average error across the entire spectrogram.
- **SNR (dB):** signal-to-noise ratio of the reconstruction; higher values indicate better quality.
- **% effectively masked:** effective percentage of hidden patches.

**Expected analysis:** as `mask_ratio` increases, the task becomes more difficult, MSE tends to rise,
and SNR tends to fall. Even with 75–80% of patches hidden, the model recovers global structures such as
harmonics (horizontal lines in the frequency domain) and the temporal envelope of the sound. Sounds with
repetitive patterns (music, events) are reconstructed better than speech, which is more unpredictable—
consistent with what is reported in the paper.

```
[ Image pending: reconstruction panels + metrics table ]
```

### Classification

The classification page shows the encoder embedding, the probabilities per class, and the class
with the highest score, along with the L2 norm of the embedding.

**Expected analysis:** Since the classification head is not trained (random linear layer), the
predictions are not reliable as actual labels. The value of this section is **demonstrative**: it shows
that the encoder produces a stable, 768-dimensional global embedding from any audio, which
is exactly what would be used as a starting point for supervised fine-tuning.

---

## 7. Conclusions

### Key Takeaways

- We understood how the same principle (masking and reconstruction) applied in BERT (text) and MAE
  (images) extends to audio via spectrograms.
- We gained a detailed understanding of the Transformer encoder–decoder workflow: patching, Q/K/V generation,
  multi-head scaled attention, and reconstruction from masked tokens.
- We saw the value of the model’s **asymmetry**: an encoder that processes only the visible portion allows for the use of
  large Transformers at a manageable cost.

### Limitations

- **Fixed resolution:** the model expects exactly 1024 × 128; audio clips of other durations are cropped or
  repeated, which introduces artifacts.
- **The output is a spectrogram, not audio:** to hear the result, a vocoder would be needed
  (e.g., Griffin-Lim or HiFi-GAN).
- **Classification without training:** The linear layer is random, so it does not reflect the actual capability
  of the encoder without fine-tuning.
- **Computational cost:** Inference runs smoothly on a GPU; on a CPU it is slower, and the checkpoint is
  ~330 MB.

### Possible Improvements

- Integrate a vocoder to play back the reconstructed audio.
- Fine-tune the encoder using a labeled dataset (e.g., ESC-50) to obtain
  accurate classification.
- Support audio of variable duration using sliding windows.
- Visualize the attention maps of the encoder heads for interpretive analysis.

---

## 8. References

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
