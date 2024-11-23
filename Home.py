# Home.py
import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from utils.document_manager import DocumentManager

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Yachani - Biblioteca Educativa",
    page_icon="📚",
    layout="wide"
)

# Inicializar el gestor de documentos
doc_manager = DocumentManager()

st.title("📚 Yachani")
st.markdown("""
### Democratizando la educación a través del conocimiento compartido

Yachani es una plataforma educativa que pone al alcance de todos recursos educativos 
de calidad mediante tecnología avanzada de procesamiento de lenguaje natural.

#### 🎯 Nuestra Misión
Hacer que el conocimiento sea accesible, organizado y fácil de encontrar para todos 
los estudiantes, sin importar su ubicación o recursos.

#### 📊 Estadísticas de la Biblioteca
""")

# Mostrar estadísticas
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Documentos", doc_manager.get_total_documents())

with col2:
    st.metric("Categorías", len(doc_manager.get_categories()))

with col3:
    st.metric("Documentos Nuevos (Hoy)", 
              doc_manager.get_new_documents_count(datetime.now()))

# Mostrar categorías populares
st.subheader("🏷️ Categorías Destacadas")
categories = doc_manager.get_popular_categories()
cols = st.columns(4)
for idx, (category, count) in enumerate(categories.items()):
    with cols[idx % 4]:
        st.button(
            f"{category} ({count} docs)",
            key=f"cat_{category}",
            use_container_width=True
        )

st.sidebar.success("Explora nuestras páginas para descubrir más.")