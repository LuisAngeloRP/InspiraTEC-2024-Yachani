# pages/3_💬_chat.py
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

# Configuración de la página
st.set_page_config(
    page_title="Chat Educativo",
    page_icon="💬",
    layout="wide"
)

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

def main():
    st.title("💬 Chat Educativo")

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

    # Layout principal
    chat_col, info_col = st.columns([3, 1])

    with info_col:
        # Información del agente activo
        st.sidebar.markdown("""
        # 🤖 Asistente Activo
        """)
        
        with st.sidebar.container():
            st.markdown(f"""
            ### {config['name']}
            - 🎭 **Rol:** {config['role']}
            - 💬 **Estilo:** {config['style']}
            - 📝 **Nivel de Detalle:** {config['detail_level']}
            """)
            
            # Documentos base
            st.markdown("### 📚 Documentos Base")
            for vs in config['vectorstores']:
                st.markdown(f"- {vs['title']}")
            
            # Configuración avanzada
            with st.expander("⚙️ Configuración Avanzada"):
                st.markdown(f"""
                - Temperature: {config['temperature']}
                - Max Tokens: {config['max_tokens']}
                - Context Window: {config['context_window']}
                """)
            
            # Gestión de historiales
            st.markdown("### 💾 Gestión de Historial")
            
            # Cargar historiales existentes
            histories = []
            history_dir = os.path.join("data", "chat_history")
            if os.path.exists(history_dir):
                for file in os.listdir(history_dir):
                    if file.startswith(f"agent_{config['name']}_") and file.endswith(".json"):
                        histories.append(file[:-5])  # Remover .json
            
            if histories:
                selected_history = st.selectbox(
                    "Cargar historial anterior",
                    options=["Actual"] + histories,
                    format_func=lambda x: f"Sesión {x.split('_')[-1]}" if x != "Actual" else "Sesión Actual"
                )
                
                if selected_history != "Actual":
                    if st.button("📂 Cargar Historial"):
                        st.session_state.messages = load_agent_history(selected_history)
                        st.rerun()
            
            # Opciones de historial
            if st.button("🗑️ Limpiar Chat"):
                st.session_state.messages = []
                if 'agent' in st.session_state:
                    del st.session_state.agent
                st.rerun()
            
            if st.button("💾 Guardar Historial"):
                if st.session_state.messages:
                    save_agent_history(agent_id, st.session_state.messages)
                    st.success("✅ Historial guardado correctamente")

    with chat_col:
        # Inicializar chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
            welcome_message = {
                "role": "assistant",
                "content": f"""¡Hola! Soy {config['name']}, tu {config['role']}.
                Estoy aquí para ayudarte con los documentos que has seleccionado.
                Para obtener mejores resultados, por favor sé específico en tus preguntas.""",
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(welcome_message)

        # Mostrar historial
        for message in st.session_state.messages:
            show_chat_message(message)

        # Input del usuario
        if prompt := st.chat_input("¿Qué deseas saber?"):
            # Agregar timestamp al mensaje
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
                                        # Realizar búsqueda
                                        docs = vs['retriever'].get_relevant_documents(query)
                                        for doc in docs:
                                            content = doc.page_content.strip()
                                            source = vs['title']
                                            
                                            # Verificar si el contenido es nuevo
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

                            st.session_state.agent = initialize_agent(
                                tools,
                                llm,
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                verbose=True,
                                max_iterations=3,
                                memory=ConversationBufferMemory(memory_key="chat_history"),
                                handle_parsing_errors=True
                            )

                        # Procesar consulta
                        response = st.session_state.agent.run(
                            f"""Actúa como {config['name']}, un {config['role']} con estilo {config['style'].lower()}.
                            
                            Consulta: {prompt}
                            
                            Instrucciones:
                            1. Usa search_documents para encontrar información relevante
                            2. Responde usando SOLO información de los documentos
                            3. Cita las fuentes usando [Documento]
                            4. Mantén un nivel de detalle {config['detail_level'].lower()}
                            5. Si no encuentras información, sugiere cómo reformular la pregunta
                            """
                        )
                        
                        # Agregar timestamp a la respuesta
                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        st.markdown(response)
                        st.session_state.messages.append(assistant_message)
                        
                        # Guardar historial automáticamente
                        save_agent_history(agent_id, st.session_state.messages)

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })

        # Mantener historial manejable
        if len(st.session_state.messages) > 15:
            st.session_state.messages = st.session_state.messages[-15:]

# Estilos CSS
st.markdown("""
<style>
    /* Mensajes del chat */
    .stChatMessage {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    
    /* Información del agente */
    .stSidebar .stMarkdown {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Timestamps */
    .stChatMessage small {
        color: #6c757d;
        font-size: 0.8em;
    }
    
    /* Contenedores */
    .stContainer {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Input del chat */
    .stChatInputContainer {
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()