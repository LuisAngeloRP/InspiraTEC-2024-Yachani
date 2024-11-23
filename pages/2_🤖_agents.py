# pages/2_🤖_agents.py
import os
import streamlit as st
from utils.document_manager import DocumentManager
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import json
from datetime import datetime

st.set_page_config(
    page_title="Gestión de Asistentes",
    page_icon="🤖",
    layout="wide"
)

def load_saved_agents():
    """Cargar agentes guardados del archivo JSON."""
    try:
        if os.path.exists("data/saved_agents.json"):
            with open("data/saved_agents.json", "r") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error al cargar agentes guardados: {str(e)}")
    return {}

def save_agent(agent_config):
    """Guardar configuración del agente."""
    try:
        agents = load_saved_agents()
        
        # Crear ID único para el agente
        agent_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Guardar información esencial
        agents[agent_id] = {
            'name': agent_config['name'],
            'role': agent_config['role'],
            'style': agent_config['style'],
            'detail_level': agent_config['detail_level'],
            'temperature': agent_config['temperature'],
            'max_tokens': agent_config['max_tokens'],
            'context_window': agent_config['context_window'],
            'docs': [{'title': vs['title'], 'hash': vs['hash']} for vs in agent_config['vectorstores']],
            'created_at': datetime.now().isoformat()
        }
        
        os.makedirs("data", exist_ok=True)
        with open("data/saved_agents.json", "w") as f:
            json.dump(agents, f, indent=2)
        
        return agent_id
    except Exception as e:
        st.error(f"Error al guardar el agente: {str(e)}")
        return None

def delete_agent(agent_id):
    """Eliminar un agente guardado."""
    try:
        agents = load_saved_agents()
        if agent_id in agents:
            del agents[agent_id]
            with open("data/saved_agents.json", "w") as f:
                json.dump(agents, f, indent=2)
            return True
    except Exception as e:
        st.error(f"Error al eliminar el agente: {str(e)}")
    return False

def load_agent_config(agent_id, doc_manager):
    """Cargar configuración completa del agente."""
    try:
        agents = load_saved_agents()
        if agent_id not in agents:
            return None
        
        saved_agent = agents[agent_id]
        
        # Inicializar vectorstores
        vectorstores = []
        for doc_info in saved_agent['docs']:
            doc = doc_manager.get_document(doc_info['hash'])
            if doc and os.path.exists(doc.get('vectorstore_path', '')):
                vectorstore = Chroma(
                    persist_directory=doc['vectorstore_path'],
                    embedding_function=OpenAIEmbeddings()
                )
                
                vectorstores.append({
                    'hash': doc['hash'],
                    'title': doc['title'],
                    'vectorstore': vectorstore,
                    'retriever': vectorstore.as_retriever(
                        search_kwargs={"k": saved_agent['context_window']}
                    )
                })
        
        # Reconstruir configuración completa
        return {
            **saved_agent,
            'vectorstores': vectorstores
        }
    except Exception as e:
        st.error(f"Error al cargar la configuración del agente: {str(e)}")
        return None

def main():

    st.title("🤖 Gestión de Asistentes")
    
    # Inicializar doc_manager
    doc_manager = DocumentManager()

    # Tabs principales
    tab_saved, tab_create = st.tabs(["📚 Asistentes Guardados", "✨ Crear Nuevo Asistente"])

    # Tab de Asistentes Guardados
    with tab_saved:
        st.subheader("Asistentes Disponibles")
        saved_agents = load_saved_agents()
        
        if not saved_agents:
            st.info("No hay asistentes guardados. Crea uno nuevo en la pestaña 'Crear Nuevo Asistente'.")
        else:
            # Mostrar agentes en grid
            for idx, (agent_id, agent) in enumerate(saved_agents.items()):
                with st.container():
                    st.markdown(f"""
                    ### {agent['name']}
                    - 🎭 **Rol:** {agent['role']}
                    - 💬 **Estilo:** {agent['style']}
                    - 📚 **Documentos:** {len(agent['docs'])}
                    - 📅 **Creado:** {datetime.fromisoformat(agent['created_at']).strftime('%d/%m/%Y %H:%M')}
                    """)
                    
                    # Documentos asociados
                    with st.expander("📄 Ver Documentos"):
                        for doc in agent['docs']:
                            st.markdown(f"- {doc['title']}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("💬 Usar", key=f"use_{agent_id}"):
                            with st.spinner("Cargando asistente..."):
                                config = load_agent_config(agent_id, doc_manager)
                                if config and config['vectorstores']:
                                    st.session_state['current_agent_config'] = config
                                    st.success(f"✅ Asistente '{agent['name']}' cargado")
                                    st.switch_page("pages/3_💬_chat.py")
                                else:
                                    st.error("❌ Error al cargar el asistente")
                    
                    with col2:
                        if st.button("🗑️ Eliminar", key=f"delete_{agent_id}"):
                            if delete_agent(agent_id):
                                st.success("Asistente eliminado correctamente")
                                st.rerun()
                    
                    st.markdown("---")

    # Tab de Crear Nuevo Asistente
    with tab_create:
        if 'selected_docs' not in st.session_state or not st.session_state.selected_docs:
            st.warning("""
            ⚠️ No hay documentos seleccionados.
            Por favor, ve al Catálogo para seleccionar los documentos base para tu asistente.
            """)
            if st.button("📚 Ir al Catálogo"):
                st.switch_page("pages/1_📚_catalog.py")
            st.stop()

        # Mostrar documentos seleccionados
        st.subheader("📚 Documentos Base")
        selected_docs_info = []

        # Grid de documentos
        cols = st.columns(3)
        for idx, doc_hash in enumerate(st.session_state.selected_docs):
            doc = doc_manager.get_document(doc_hash)
            if doc:
                with cols[idx % 3]:
                    st.markdown(f"""
                    ### 📄 {doc['title']}
                    - **Categoría:** {doc.get('category', 'No especificada')}
                    - **Tipo:** {doc.get('type', 'No especificado')}
                    - **Nivel:** {doc.get('level', 'No especificado')}
                    - **Páginas:** {doc.get('pages', '0')}
                    """)
                    
                    if st.button("❌ Remover", key=f"remove_{doc_hash}"):
                        st.session_state.selected_docs.remove(doc_hash)
                        st.rerun()
                    
                    selected_docs_info.append(doc)

        # Formulario de configuración
        st.markdown("---")
        st.subheader("⚙️ Configuración del Asistente")

        with st.form("agent_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                agent_name = st.text_input(
                    "Nombre del Asistente",
                    placeholder="Ej: Tutor de Matemáticas",
                    help="Un nombre descriptivo para tu asistente"
                )
                
                agent_role = st.selectbox(
                    "Rol del Asistente",
                    options=[
                        "Tutor Personal",
                        "Asistente de Investigación",
                        "Profesor",
                        "Mentor",
                        "Consultor Especializado"
                    ],
                    help="Define cómo se comportará el asistente"
                )
            
            with col2:
                communication_style = st.select_slider(
                    "Estilo de Comunicación",
                    options=["Formal", "Balanceado", "Casual"],
                    value="Balanceado",
                    help="Define el tono de las respuestas"
                )
                
                detail_level = st.select_slider(
                    "Nivel de Detalle",
                    options=["Conciso", "Moderado", "Detallado"],
                    value="Moderado",
                    help="Define qué tan extensas serán las respuestas"
                )

            with st.expander("🔧 Opciones Avanzadas"):
                col3, col4 = st.columns(2)
                
                with col3:
                    temperature = st.slider(
                        "Creatividad (Temperature)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="Mayor valor = respuestas más creativas"
                    )
                    
                    context_window = st.slider(
                        "Contexto (k)",
                        min_value=1,
                        max_value=10,
                        value=5,
                        help="Número de fragmentos de contexto a usar"
                    )
                
                with col4:
                    max_tokens = st.select_slider(
                        "Longitud Máxima",
                        options=[512, 1024, 2048, 4096],
                        value=2048,
                        help="Longitud máxima de las respuestas"
                    )
            
            submitted = st.form_submit_button("🚀 Crear Asistente", use_container_width=True)

        if submitted:
            if not agent_name:
                st.error("❌ Por favor, asigna un nombre a tu asistente.")
                st.stop()
            
            if not selected_docs_info:
                st.error("❌ Necesitas seleccionar al menos un documento base.")
                st.stop()

            with st.spinner("⚙️ Configurando tu asistente..."):
                try:
                    # Inicializar vectorstores
                    vectorstores = []
                    for doc in selected_docs_info:
                        vectorstore_path = doc.get('vectorstore_path')
                        if vectorstore_path and os.path.exists(vectorstore_path):
                            vectorstore = Chroma(
                                persist_directory=vectorstore_path,
                                embedding_function=OpenAIEmbeddings()
                            )
                            
                            vectorstores.append({
                                'hash': doc['hash'],
                                'title': doc['title'],
                                'vectorstore': vectorstore,
                                'retriever': vectorstore.as_retriever(
                                    search_kwargs={"k": context_window}
                                )
                            })
                        else:
                            st.warning(f"⚠️ No se encontró el vectorstore para {doc['title']}")

                    if vectorstores:
                        # Crear configuración
                        agent_config = {
                            'name': agent_name,
                            'role': agent_role,
                            'style': communication_style,
                            'detail_level': detail_level,
                            'temperature': temperature,
                            'max_tokens': max_tokens,
                            'context_window': context_window,
                            'vectorstores': vectorstores
                        }
                        
                        # Guardar agente
                        agent_id = save_agent(agent_config)
                        
                        if agent_id:
                            # Guardar en session state
                            st.session_state['current_agent_config'] = agent_config
                            
                            st.success(f"""
                            ✅ Asistente "{agent_name}" creado y guardado exitosamente:
                            - 📚 {len(vectorstores)} documentos base cargados
                            - 🎭 Rol: {agent_role}
                            - 💬 Estilo: {communication_style}
                            """)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📝 Crear Otro Asistente"):
                                    st.session_state.selected_docs = []
                                    st.rerun()
                            with col2:
                                if st.button("💬 Ir al Chat"):
                                    st.switch_page("pages/3_💬_chat.py")
                        else:
                            st.error("❌ Error al guardar el asistente")
                    else:
                        st.error("❌ No se pudo cargar ningún vectorstore.")

                except Exception as e:
                    st.error(f"❌ Error al configurar el asistente: {str(e)}")

if __name__ == "__main__":
    main()