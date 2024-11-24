import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from typing import List, Dict
import re
import json
import os
from datetime import datetime
import base64

# Configuración de la página
st.set_page_config(
    page_title="Vista Documento + Chat",
    page_icon="📑",
    layout="wide"
)

# Funciones auxiliares del chat (reutilizadas del chat.py)
def load_agent_history(agent_id: str) -> List[Dict]:
    """Carga el historial de conversaciones de un agente específico."""
    history_path = os.path.join("data", "chat_history", f"{agent_id}.json")
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_agent_history(agent_id: str, messages: List[Dict]):
    """Guarda el historial de conversaciones de un agente."""
    os.makedirs(os.path.join("data", "chat_history"), exist_ok=True)
    history_path = os.path.join("data", "chat_history", f"{agent_id}.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def format_timestamp(timestamp: str) -> str:
    """Formatea un timestamp para mostrar."""
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%d/%m/%Y %H:%M")

def show_chat_message(message: Dict, show_timestamp: bool = True):
    """Muestra un mensaje del chat con formato mejorado."""
    with st.chat_message(message["role"]):
        if show_timestamp and "timestamp" in message:
            st.caption(format_timestamp(message["timestamp"]))
        st.markdown(message["content"])

def get_agent_id(config: Dict) -> str:
    """Genera un ID único para el agente basado en su configuración."""
    return f"agent_{config['name']}_{datetime.now().strftime('%Y%m%d')}"

def get_recent_history(messages: List[Dict], max_messages: int = 5) -> str:
    """Obtiene el historial reciente de mensajes formateado."""
    recent_messages = messages[-max_messages:] if messages else []
    formatted_history = []
    for msg in recent_messages:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {msg['content']}")
    return "\n".join(formatted_history)

def display_pdf(pdf_path: str):
    """Muestra un PDF en el iframe."""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def extract_pdf_content(pdf_path: str) -> List[Dict]:
    """
    Extrae el contenido del PDF página por página y lo devuelve como una lista de diccionarios.
    Cada diccionario contiene el número de página y su contenido.
    """
    import fitz  # PyMuPDF
    pages_content = []
    
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            content = page.get_text("text").strip()
            # Extraer los primeros caracteres para el índice
            preview = content[:100] + "..." if len(content) > 100 else content
            
            pages_content.append({
                'page_num': page_num + 1,
                'content': content,
                'preview': preview
            })
        return pages_content
    except Exception as e:
        st.error(f"Error al extraer contenido del PDF: {str(e)}")
        return []

def display_content_viewer(pdf_path: str):
    """
    Muestra el contenido del PDF en un formato de texto con navegación por páginas.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        st.error("El archivo no se encuentra disponible.")
        st.markdown(f"Ruta esperada: {pdf_path}")
        return

    # Extraer contenido del PDF si no está en caché
    cache_key = f"pdf_content_{pdf_path}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = extract_pdf_content(pdf_path)
    
    pages_content = st.session_state[cache_key]
    
    if not pages_content:
        st.error("No se pudo extraer el contenido del documento.")
        return

    # Layout con dos columnas: índice y contenido
    index_col, content_col = st.columns([0.3, 0.7])
    
    with index_col:
        st.markdown("### 📑 Índice")
        # Crear selectbox para navegación
        page_options = [f"Página {page['page_num']}" for page in pages_content]
        selected_page = st.radio(
            "Seleccionar página",
            options=range(len(pages_content)),
            format_func=lambda x: f"Página {pages_content[x]['page_num']}\n{pages_content[x]['preview'][:50]}...",
            label_visibility="collapsed"
        )

    with content_col:
        st.markdown("### 📄 Contenido")
        # Mostrar contenido de la página seleccionada
        page_data = pages_content[selected_page]
        
        # Contenedor con estilo para el contenido
        st.markdown(f"""
        <div class="content-box">
            <div class="page-header">
                <h4>Página {page_data['page_num']} de {len(pages_content)}</h4>
            </div>
            <div class="page-content">
                {page_data['content'].replace('\n', '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_document_info(vectorstores: List[Dict]) -> List[Dict]:
    """
    Extrae solo la información necesaria de los vectorstores para el selector.
    Asume que el PDF está en processed_docs/[AGENTE]/[AGENTE].pdf
    """
    docs_info = []
    for vs in vectorstores:
        # El hash es el nombre del agente/carpeta
        agent_folder = vs.get('hash', '')
        # Construir la ruta al PDF usando el mismo nombre de carpeta para el archivo
        pdf_path = os.path.join('processed_docs', agent_folder, f"{agent_folder}.pdf")
        
        docs_info.append({
            'title': vs['title'],
            'path': pdf_path,
            'agent_folder': agent_folder
        })
    
    return docs_info

def main():
    st.title("📑 Vista Documento + Chat")

    # Verificar configuración del agente
    if 'current_agent_config' not in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.warning("""
            ⚠️ No hay un asistente configurado.
            Por favor, configura tu asistente primero.
            """)
            if st.button("🤖 Configurar Asistente", use_container_width=True):
                st.switch_page("pages/2_🤖_agents.py")
        st.stop()

    config = st.session_state.current_agent_config
    agent_id = get_agent_id(config)

    # Layout principal con dos columnas
    doc_col, chat_col = st.columns([1.2, 0.8])

    # Columna del documento
    with doc_col:
        st.markdown("### 📄 Material de Estudio")
        
        try:
            # Obtener información de documentos
            docs_info = get_document_info(config['vectorstores'])
            
            if docs_info:
                # Selector de documento
                selected_doc = st.selectbox(
                    "Seleccionar documento",
                    options=docs_info,
                    format_func=lambda x: x['title']
                )
                
                if selected_doc:
                    # Verificar que el archivo existe
                    if os.path.exists(selected_doc['path']):
                        display_content_viewer(selected_doc['path'])
                    else:
                        st.error(f"""
                        No se encontró el archivo PDF.
                        Ruta esperada: {selected_doc['path']}
                        Asegúrate de que existe el archivo: {selected_doc['agent_folder']}.pdf
                        dentro de la carpeta: processed_docs/{selected_doc['agent_folder']}/
                        """)
            else:
                st.warning("No hay documentos disponibles")
                
        except Exception as e:
            st.error(f"Error al cargar los documentos: {str(e)}")

    # ... (resto del código del chat permanece igual)

    # Columna del chat
    with chat_col:
        st.markdown("### 💬 Asistente IA")
        
        # Información del agente
        with st.expander("ℹ️ Información del Asistente"):
            st.markdown(f"""
            **{config['name']}**
            - 🎭 Rol: {config['role']}
            - 💬 Estilo: {config['style']}
            - 📝 Nivel: {config['detail_level']}
            """)

        # Chat container con scroll
        chat_container = st.container()
        with chat_container:
            # Inicializar mensajes
            if "messages" not in st.session_state:
                st.session_state.messages = []
                welcome_message = {
                    "role": "assistant",
                    "content": f"""¡Hola! Soy {config['name']}, tu {config['role']}.
                    Estoy aquí para ayudarte con el material que estás revisando.
                    Puedes preguntarme cualquier cosa sobre el documento.""",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(welcome_message)

            # Mostrar mensajes
            for message in st.session_state.messages:
                show_chat_message(message)

        # Input del usuario
        if prompt := st.chat_input("¿Qué deseas saber sobre el material?"):
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            show_chat_message(user_message)

            with st.chat_message("assistant"):
                with st.spinner(f"💭 {config['name']} está pensando..."):
                    try:
                        # Inicializar agente si no existe
                        if "agent" not in st.session_state:
                            llm = ChatOpenAI(
                                temperature=config['temperature'],
                                model="gpt-4-0125-preview",
                                max_tokens=config['max_tokens']
                            )

                            def search_documents(query: str) -> str:
                                """Buscar información en los documentos base."""
                                try:
                                    results = []
                                    for vs in config['vectorstores']:
                                        docs = vs['retriever'].get_relevant_documents(query)
                                        for doc in docs:
                                            content = doc.page_content.strip()
                                            source = vs['title']
                                            if content not in [r.split(']:')[1].strip() for r in results]:
                                                results.append(f"[{source}]: {content}")
                                    if results:
                                        return "\n\n".join(results[:config['context_window']])
                                    return "No encontré información específica. ¿Podrías reformular la pregunta?"
                                except Exception as e:
                                    return f"Error al buscar: {str(e)}"

                            tools = [
                                Tool(
                                    name="search_documents",
                                    func=search_documents,
                                    description="Busca información en los documentos base."
                                )
                            ]

                            memory = ConversationBufferMemory(
                                memory_key="chat_history",
                                return_messages=True
                            )

                            st.session_state.agent = initialize_agent(
                                tools,
                                llm,
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                verbose=True,
                                max_iterations=3,
                                memory=memory,
                                handle_parsing_errors=True
                            )

                        # Procesar consulta
                        recent_history = get_recent_history(st.session_state.messages)
                        prompt_text = f"""Actúa como {config['name']}, un {config['role']} con estilo {config['style'].lower()}.
                        
                        Historial reciente:
                        {recent_history}
                        
                        Consulta actual: {prompt}
                        
                        Instrucciones:
                        1. Usa search_documents para buscar información relevante
                        2. Responde usando SOLO información de los documentos
                        3. Cita las fuentes usando [Documento]
                        4. Mantén un nivel de detalle {config['detail_level'].lower()}
                        5. Si no encuentras información, sugiere cómo reformular la pregunta
                        """
                        
                        response = st.session_state.agent.run(prompt_text)
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        st.markdown(response)
                        st.session_state.messages.append(assistant_message)
                        save_agent_history(agent_id, st.session_state.messages)

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })

# Estilos CSS personalizados
st.markdown("""
<style>
    /* Contenedor de índice */
    .index-container {
        border-right: 1px solid #e0e0e0;
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding-right: 1rem;
    }

    /* Contenedor de contenido */
    .content-box {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        height: calc(100vh - 250px);
        overflow-y: auto;
    }

    /* Encabezado de página */
    .page-header {
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
    }

    /* Contenido de página */
    .page-content {
        font-size: 1rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    /* Radio buttons del índice */
    .stRadio > label {
        font-size: 0.9rem;
        padding: 0.5rem;
        border-radius: 5px;
    }

    .stRadio > div[role="radiogroup"] > div {
        margin-bottom: 0.5rem;
    }

    /* Scroll personalizado */
    .content-box::-webkit-scrollbar {
        width: 8px;
    }

    .content-box::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    .content-box::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    .content-box::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()