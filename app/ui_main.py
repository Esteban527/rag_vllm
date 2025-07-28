import streamlit as st
import os
import openai
from dotenv import load_dotenv
import requests

# Importar lógica RAG del módulo local app/rag_logic.py.
try:
    from rag_logic import (
        get_all_documents, # Para cargar datos para Lunr y el mapa de documentos.
        hybrid_search_orchestrator      # Orquestador principal de la búsqueda híbrida.
    )
except ImportError as e:
    st.error(f"Error: No se pudo importar 'rag_logic.py': {e}.")
    st.stop() 

load_dotenv(override=True)


VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://192.168.1.10:8002")
VLLM_EMBED = os.environ.get("VLLM_EMBED","http://192.168.1.10:8003/embed")
VLLM_MODEL_GENERATION = os.environ.get("VLLM_GENERATION") 
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")

client = openai.OpenAI(
    base_url=VLLM_ENDPOINT + "/v1",
    api_key="not-needed"  # vLLM no requiere autenticación por defecto
)
# Construye rutas a archivos.
APP_DIR = os.path.dirname(__file__)
PROMPT_DIR = os.path.join(APP_DIR, "prompts")
SYSTEM_PROMPT_FILE = os.path.join(PROMPT_DIR, "wifi.txt")

# Número de documentos a recuperar para el contexto RAG.
NUM_DOCS_FOR_CONTEXT = 3

# Verifica que las variables de entorno críticas estén definidas.
critical_configs = {
    "VLLM_ENDPOINT": VLLM_ENDPOINT,
    "VLLM_MODEL_GENERATION": VLLM_MODEL_GENERATION,
    "DB_CONNECTION_STRING": DB_CONNECTION_STRING
}
missing_configs = [key for key, value in critical_configs.items() if not value]

# --- Funciones Cacheadas ---
# Usan el caché de Streamlit para evitar recargar recursos costosos en cada interacción (modelos, datos) en cada interacción.
@st.cache_resource
def get_vllm_client_cached():
    """Inicializa y cachea la configuración del endpoint vLLM."""
    if missing_configs:
        return None
    return VLLM_ENDPOINT

@st.cache_data(show_spinner="Cargando base de conocimiento...")
def cached_fetch_all_documents(_db_conn_string_ref):
    """Recupera y cachea documentos de la BD para Lunr y el mapa de documentos."""
    if not _db_conn_string_ref: 
        st.warning("No se ha definido una cadena de conexión a la base de datos. No se cargarán documentos.")
        return []
    try:
        return get_all_documents(_db_conn_string_ref)
    except Exception:
        return []

@st.cache_resource(show_spinner="Preparando herramientas de búsqueda...")
def load_search_tools_cached(_documents_list_ref):
    """Crea y cachea el índice Lunr y el mapa de documentos por ID."""
    from lunr import lunr 
    if not _documents_list_ref:
        st.warning("No se han cargado documentos. No se creará el índice de búsqueda.") 
        return None, {}
    documents_by_id_map = {doc["id"]: doc for doc in _documents_list_ref}
    valid_docs_for_lunr = [{"id": doc["id"], "text": doc["text"]} for doc in _documents_list_ref if doc.get("text")]
    if not valid_docs_for_lunr: return None, documents_by_id_map
    try:
        idx = lunr(ref="id", fields=["text"], documents=valid_docs_for_lunr)
        return idx, documents_by_id_map
    except Exception:
        return None, documents_by_id_map

@st.cache_resource(show_spinner="Cargando modelo de re-ranking ")
def get_cross_encoder_model_cached_simplified():
    """Carga y cachea el modelo CrossEncoder."""
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        st.warning("No se pudo cargar el modelo CrossEncoder. Usando modelo por defecto.")
        return None

@st.cache_data(show_spinner="Cargando instrucciones del agente...")
def load_system_prompt_cached(_prompt_file_path):
    """Carga y cachea el prompt del sistema desde un archivo."""
    try:
        with open(_prompt_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        st.warning(f"No se pudo cargar el prompt del sistema desde '{_prompt_file_path}'. Usando prompt por defecto.")
        return "Eres un asistente IA que responde preguntas basándose en un contexto."

# --- Inicialización y Carga de Recursos ---
st.set_page_config(page_title="Agente WiFi", layout="centered")
st.title("🤖 Agente WiFi")

if missing_configs: # Si faltan configuraciones críticas, detiene.
    st.error(f"Error: Faltan configuraciones en .env: {', '.join(missing_configs)}.")
    st.stop()

# Carga los recursos principales para la aplicación.
with st.spinner("Iniciando agente... Por favor, espera."):
    VLLM_client = get_vllm_client_cached()
    documents_list_from_db = cached_fetch_all_documents(DB_CONNECTION_STRING)
    lunr_idx, docs_map = load_search_tools_cached(documents_list_from_db)
    cross_encoder_model = get_cross_encoder_model_cached_simplified() 
    system_message = load_system_prompt_cached(SYSTEM_PROMPT_FILE)

# Verifica si los recursos críticos se cargaron.
if not VLLM_client or not documents_list_from_db or not lunr_idx:
    st.error("No se pudieron cargar los recursos necesarios. La aplicación no puede continuar.")
    st.stop()

st.success(f"Agente listo.")

# --- Gestión del Historial de Chat ---
# Usa st.session_state para mantener el historial durante la sesión del usuario.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

# --- Interfaz de Chat Principal ---
for role, text in st.session_state.chat_history:
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(text)

# Campo de entrada para la pregunta del usuario.
if user_question := st.chat_input("Escribe tu pregunta sobre WiFi..."):
    # Añade la pregunta del usuario al historial y la muestra.
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_question)

    # Placeholder para la respuesta del asistente (para efecto de streaming).
    assistant_response_placeholder = st.chat_message("assistant", avatar="🤖").empty()
    current_response_text = "" # Acumula la respuesta en streaming.
    
    # Realiza la búsqueda RAG y genera la respuesta.
    with st.spinner("Buscando información y generando respuesta..."):
        try:
            retrieved_docs = hybrid_search_orchestrator(
                query=user_question,
                limit=NUM_DOCS_FOR_CONTEXT,
                lunr_search_idx=lunr_idx,
                docs_by_id_map=docs_map,
                embeddings_url=VLLM_EMBED,
                cross_enc_model_inst=cross_encoder_model, 
                db_conn_str_param=DB_CONNECTION_STRING,
            )
        except Exception as e:
            st.error(f"Error durante la búsqueda de documentos: {e}")
            retrieved_docs = []

        # Prepara el contexto RAG para la pregunta actual.
        context_for_current_question_only = ""
        if retrieved_docs:
            context_for_current_question_only = "\n\n---\n\n".join(
                [f"Contenido: {doc.get('text', '')}" for doc in retrieved_docs]
            )

        # Construye la lista de mensajes para el LLM, incluyendo el historial.
        messages_for_llm = [{"role": "system", "content": system_message}]
        for role, text_content in st.session_state.chat_history:
            if role == "user" and text_content == user_question: # Pregunta actual del usuario
                if retrieved_docs: # Si hay contexto RAG para esta pregunta
                    prompt_with_rag_context = f"Pregunta del usuario: {user_question}\n\nBasándote en las siguientes fuentes, responde la pregunta directamente. Si la respuesta no está en las fuentes, indica que no tienes información suficiente.\nFuentes:\n{context_for_current_question_only}"
                    messages_for_llm.append({"role": "user", "content": prompt_with_rag_context})
                else: # Pregunta actual sin contexto RAG
                    st.warning("No se encontró información específica sobre este tema en la documentación disponible.")
                    messages_for_llm.append({"role": "user", "content": f"Pregunta del usuario: {user_question}\n\nNo se encontró información específica sobre este tema en la documentación disponible. Responde indicando que no tienes información suficiente sobre este aspecto específico."})
            else: # Turnos anteriores (preguntas de usuario pasadas o respuestas del asistente)
                messages_for_llm.append({"role": role, "content": text_content})
        
        # Limita la longitud del historial enviado al LLM.
        MAX_CONVERSATION_HISTORY_FOR_LLM = 10 
        if len(messages_for_llm) > (MAX_CONVERSATION_HISTORY_FOR_LLM + 1):
            messages_for_llm = [messages_for_llm[0]] + messages_for_llm[-(MAX_CONVERSATION_HISTORY_FOR_LLM):]

        with st.expander("Ver Documentos y Contexto Enviado al LLM", expanded=False):
            st.markdown("---")
            # Itera sobre la lista 'retrieved_docs' que contiene los documentos recuperados.
            for i, doc in enumerate(retrieved_docs):
                # Muestra el ID del documento para una fácil referencia.
                st.markdown(f"**Documento {i+1} (ID: {doc.get('id', 'N/A')})**")
                # Muestra los primeros 300 caracteres del contenido del documento.
                st.caption(doc.get('text', 'Texto no disponible') + "...")
           
            st.markdown("---")
           
            # Muestra el prompt completo que se envía al rol 'user' del LLM.
            # Usamos 'messages_for_llm' que es la variable string que contiene la pregunta y el contexto.
            st.text_area(
                "Prompt Final Enviado al LLM (en el mensaje del usuario):",
                value=messages_for_llm, # 'value' debe ser un string
                height=250,
                key=f"context_debug_{len(st.session_state.chat_history)}" )

        # Llama al LLM de vLLM con el historial y contexto usando requests
        try:
            url = f"{VLLM_ENDPOINT}:8002/v1/chat/completions"
            payload = {
                "model": VLLM_MODEL_GENERATION,
                "messages": messages_for_llm,
                "temperature": 0.2,
                "stream": True,
                "max_tokens": 800,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3
            }
            with requests.post(url, json=payload, stream=True, timeout=120) as response:
                response.raise_for_status()
                current_response_text = ""
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    import json
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        current_response_text += content
                        assistant_response_placeholder.markdown(current_response_text + "▌")
                assistant_response_placeholder.markdown(current_response_text)
                st.session_state.chat_history.append(("assistant", current_response_text))
        except Exception as e:
            error_msg = f"Error al generar respuesta con LLM: {e}"
            st.error(error_msg)
            current_response_text = "Lo siento, tuve un problema al generar la respuesta."
            assistant_response_placeholder.markdown(current_response_text)
            st.session_state.chat_history.append(("assistant", current_response_text))

##Comandos para ejecutar la aplicación:
# python -m venv venv    
# venv\Scripts\activate   
# pip install -r requirements.txt
# streamlit run app/ui_main.py 
# python scripts/ingest_data.py
