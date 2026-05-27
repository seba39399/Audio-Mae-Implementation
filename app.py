"""
AudioMAE - Masked Autoencoder for Audio
Interfaz principal de Streamlit
"""

import streamlit as st

st.set_page_config(
    page_title="AudioMAE Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.sidebar import render_sidebar
from ui.page_reconstruction import render_reconstruction_page
from ui.page_classification import render_classification_page
from ui.page_architecture import render_architecture_page

def main():
    # Sidebar: selección de página y configuración global
    page, config = render_sidebar()

    if page == "🔄 Reconstrucción (Generación)":
        render_reconstruction_page(config)
    elif page == "🏷️ Clasificación":
        render_classification_page(config)
    elif page == "🧠 Arquitectura del Modelo":
        render_architecture_page()

if __name__ == "__main__":
    main()
