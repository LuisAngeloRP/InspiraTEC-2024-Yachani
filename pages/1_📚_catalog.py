# pages/1_📚_catalog.py
import streamlit as st
from utils.document_manager import DocumentManager
import os
from datetime import datetime
import base64

st.set_page_config(
        page_title="Catálogo de Documentos",
        page_icon="📚",
        layout="wide"
)

def format_date(date_str):
    """Formatea la fecha ISO a un formato más legible."""
    try:
        date = datetime.fromisoformat(date_str)
        return date.strftime("%d/%m/%Y %H:%M")
    except:
        return "Fecha no disponible"

def get_safe_value(doc, key, default="No disponible"):
    """Obtiene un valor de forma segura del diccionario."""
    return doc.get(key, default)

def format_file_size(size_in_bytes):
    """Formatea el tamaño del archivo."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} GB"

def create_download_link(file_path: str, link_text: str):
    """Crea un link de descarga para un archivo."""
    try:
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        filename = os.path.basename(file_path)
        mime_type = "application/octet-stream"
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="download-link">{link_text}</a>'
    except Exception:
        return None

def show_document_details(doc, is_full_view=True):
    """Muestra los detalles del documento con manejo seguro de campos."""
    base_info = f"""
    #### 📄 {get_safe_value(doc, 'title')}
    - **Tipo:** {get_safe_value(doc, 'type')}
    - **Nivel:** {get_safe_value(doc, 'level')}
    """
    
    if is_full_view:
        # Calcular tamaño del archivo si existe
        file_path = get_safe_value(doc, 'original_path')
        file_size = "No disponible"
        if file_path and os.path.exists(file_path):
            file_size = format_file_size(os.path.getsize(file_path))

        full_info = f"""
        - **Autor:** {get_safe_value(doc, 'author')}
        - **Idioma:** {get_safe_value(doc, 'language')}
        - **Año:** {get_safe_value(doc, 'year')}
        - **Páginas:** {get_safe_value(doc, 'pages', '0')}
        - **Fragmentos:** {get_safe_value(doc, 'chunks', '0')}
        - **Tamaño:** {file_size}
        - **Fecha de Carga:** {format_date(get_safe_value(doc, 'processed_date'))}
        
        **Descripción:**  
        {get_safe_value(doc, 'description')}
        
        **Etiquetas:**  
        {', '.join(get_safe_value(doc, 'tags', ['Sin etiquetas']))}
        """
        return base_info + full_info
    return base_info

def main():

    st.title("📚 Catálogo de Documentos")

    doc_manager = DocumentManager()

    # Layout de dos columnas principales
    col_catalog, col_search = st.columns([2, 1])

    with col_search:
        st.subheader("🔍 Buscador")
        
        # Búsqueda por texto
        search_query = st.text_input(
            "Buscar",
            placeholder="Título, descripción, etiquetas...",
            help="Busca en todos los campos del documento"
        )
        
        st.markdown("### 📋 Filtros")
        
        # Filtros
        categories = doc_manager.categories["categories"]
        selected_category = st.selectbox(
            "Categoría",
            ["Todas"] + list(categories.keys())
        )
        
        selected_type = st.selectbox(
            "Tipo de Documento",
            ["Todos"] + doc_manager.get_document_types()
        )
        
        selected_level = st.selectbox(
            "Nivel",
            ["Todos"] + doc_manager.get_difficulty_levels()
        )
        
        # Filtros adicionales
        with st.expander("🔍 Filtros Avanzados"):
            selected_language = st.selectbox(
                "Idioma",
                ["Todos", "Español", "Inglés", "Francés", "Alemán"]
            )
            
            year_range = st.slider(
                "Año de Publicación",
                min_value=1900,
                max_value=2024,
                value=(1900, 2024)
            )
        
        # Botón de búsqueda
        search_clicked = st.button("🔍 Buscar", use_container_width=True)
        
        # Mostrar documentos seleccionados
        if 'selected_docs' in st.session_state and st.session_state.selected_docs:
            st.markdown("---")
            st.markdown("### 📑 Documentos Seleccionados")
            
            total_pages = 0
            total_chunks = 0
            
            for doc_hash in st.session_state.selected_docs:
                doc = doc_manager.get_document(doc_hash)
                if doc:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{get_safe_value(doc, 'title')}**")
                            st.caption(f"{get_safe_value(doc, 'type')} · {get_safe_value(doc, 'level')}")
                        with col2:
                            if st.button("❌", key=f"remove_{doc_hash}"):
                                st.session_state.selected_docs.remove(doc_hash)
                                st.rerun()
                    
                    total_pages += int(get_safe_value(doc, 'pages', 0))
                    total_chunks += int(get_safe_value(doc, 'chunks', 0))
            
            st.markdown(f"""
            **Resumen:**
            - 📚 {len(st.session_state.selected_docs)} documentos
            - 📄 {total_pages} páginas totales
            - 📦 {total_chunks} fragmentos
            """)
            
            if st.button("🤖 Crear Asistente", use_container_width=True):
                st.switch_page("pages/2_🤖_agents.py")

    with col_catalog:
        tab_all, tab_search = st.tabs(["📚 Biblioteca", "🔍 Resultados"])
        
        with tab_all:
            all_documents = list(doc_manager.metadata.values())
            if all_documents:
                # Ordenar por fecha
                all_documents.sort(
                    key=lambda x: get_safe_value(x, 'processed_date', ''),
                    reverse=True
                )
                
                # Opciones de visualización
                view_option = st.radio(
                    "Vista:",
                    ["Lista", "Grid"],
                    horizontal=True
                )
                
                if view_option == "Grid":
                    # Vista en grid
                    cols = st.columns(3)
                    for idx, doc in enumerate(all_documents):
                        with cols[idx % 3]:
                            with st.container():
                                # Mostrar preview si existe
                                preview_path = get_safe_value(doc, 'preview_path')
                                if preview_path and os.path.exists(preview_path):
                                    st.image(preview_path, use_column_width=True)
                                
                                st.markdown(f"""
                                #### 📄 {get_safe_value(doc, 'title')}
                                - {get_safe_value(doc, 'type')}
                                - {get_safe_value(doc, 'level')}
                                """)
                                
                                # Selección y acciones
                                col1, col2 = st.columns(2)
                                with col1:
                                    is_selected = st.checkbox(
                                        "Seleccionar",
                                        key=f"select_grid_{doc['hash']}",
                                        value=doc['hash'] in st.session_state.get('selected_docs', [])
                                    )
                                with col2:
                                    if st.button("Ver más", key=f"more_{doc['hash']}"):
                                        st.session_state.selected_doc = doc['hash']
                                
                                if is_selected:
                                    if 'selected_docs' not in st.session_state:
                                        st.session_state.selected_docs = []
                                    if doc['hash'] not in st.session_state.selected_docs:
                                        st.session_state.selected_docs.append(doc['hash'])
                                else:
                                    if 'selected_docs' in st.session_state and doc['hash'] in st.session_state.selected_docs:
                                        st.session_state.selected_docs.remove(doc['hash'])
                else:
                    # Vista en lista
                    for doc in all_documents:
                        with st.expander(f"📄 {get_safe_value(doc, 'title')}", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(show_document_details(doc))
                                
                                # Agregar link de descarga si existe el archivo
                                original_path = get_safe_value(doc, 'original_path')
                                if original_path and os.path.exists(original_path):
                                    st.markdown(
                                        create_download_link(original_path, "📥 Descargar documento"),
                                        unsafe_allow_html=True
                                    )
                            
                            with col2:
                                # Preview
                                preview_path = get_safe_value(doc, 'preview_path')
                                if preview_path and os.path.exists(preview_path):
                                    st.image(preview_path, use_column_width=True)
                                
                                # Selección
                                is_selected = st.checkbox(
                                    "Seleccionar",
                                    key=f"select_list_{doc['hash']}",
                                    value=doc['hash'] in st.session_state.get('selected_docs', [])
                                )
                                
                                if is_selected:
                                    if 'selected_docs' not in st.session_state:
                                        st.session_state.selected_docs = []
                                    if doc['hash'] not in st.session_state.selected_docs:
                                        st.session_state.selected_docs.append(doc['hash'])
                                else:
                                    if 'selected_docs' in st.session_state and doc['hash'] in st.session_state.selected_docs:
                                        st.session_state.selected_docs.remove(doc['hash'])
                                
                                # Estadísticas
                                st.markdown("### 📊 Stats")
                                st.markdown(f"""
                                - 📄 {get_safe_value(doc, 'pages', '0')} páginas
                                - 📦 {get_safe_value(doc, 'chunks', '0')} fragmentos
                                """)
            else:
                st.info("No hay documentos en el catálogo. Ve a la sección de carga para agregar documentos.")

        # Tab de búsqueda
        with tab_search:
            if search_clicked or search_query:
                filters = {
                    "category": selected_category,
                    "type": selected_type,
                    "level": selected_level,
                    "language": selected_language if selected_language != "Todos" else None,
                    "year_range": year_range
                }
                
                search_results = doc_manager.search_documents(search_query, filters)
                
                if search_results:
                    st.markdown(f"### 🔍 Resultados ({len(search_results)})")
                    
                    for doc in search_results:
                        with st.expander(f"📄 {get_safe_value(doc, 'title')}", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(show_document_details(doc, False))
                                
                                # Link de descarga
                                original_path = get_safe_value(doc, 'original_path')
                                if original_path and os.path.exists(original_path):
                                    st.markdown(
                                        create_download_link(original_path, "📥 Descargar"),
                                        unsafe_allow_html=True
                                    )
                            
                            with col2:
                                # Preview
                                preview_path = get_safe_value(doc, 'preview_path')
                                if preview_path and os.path.exists(preview_path):
                                    st.image(preview_path, use_column_width=True)
                                
                                # Selección
                                is_selected = st.checkbox(
                                    "Seleccionar",
                                    key=f"select_search_{doc['hash']}",
                                    value=doc['hash'] in st.session_state.get('selected_docs', [])
                                )
                                
                                if is_selected:
                                    if 'selected_docs' not in st.session_state:
                                        st.session_state.selected_docs = []
                                    if doc['hash'] not in st.session_state.selected_docs:
                                        st.session_state.selected_docs.append(doc['hash'])
                                else:
                                    if 'selected_docs' in st.session_state and doc['hash'] in st.session_state.selected_docs:
                                        st.session_state.selected_docs.remove(doc['hash'])
                else:
                    st.info("No se encontraron documentos que coincidan con los criterios de búsqueda.")


def render_badges(doc):
    """Renderiza badges con los metadatos del documento."""
    return f"""
    <div class="badge-container">
        <span class="badge badge-category">{get_safe_value(doc, 'category')}</span>
        <span class="badge badge-type">{get_safe_value(doc, 'type')}</span>
        <span class="badge badge-level">{get_safe_value(doc, 'level')}</span>
        <span class="badge badge-language">{get_safe_value(doc, 'language')}</span>
    </div>
    """

def render_document_card(doc, preview_path=None):
    """Renderiza una tarjeta de documento para la vista grid."""
    return f"""
    <div class="document-grid">
        {'<img src="data:image/png;base64,' + encode_image(preview_path) + '" class="preview-image" />' if preview_path else ''}
        <h4>{get_safe_value(doc, 'title')}</h4>
        {render_badges(doc)}
        <div class="stats-container">
            <div class="tooltip">📄 {get_safe_value(doc, 'pages', '0')} páginas
                <span class="tooltiptext">Número total de páginas</span>
            </div>
            <div class="tooltip">📦 {get_safe_value(doc, 'chunks', '0')} fragmentos
                <span class="tooltiptext">Fragmentos para procesamiento</span>
            </div>
        </div>
    </div>
    """

def encode_image(image_path):
    """Codifica una imagen en base64 para mostrarla en HTML."""
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

if __name__ == "__main__":
    main()