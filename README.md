# 🤖 Sistema RAG Híbrido con vLLM

Un sistema avanzado de Retrieval-Augmented Generation (RAG) que combina búsqueda vectorial y textual para proporcionar respuestas precisas basadas en documentos. Especializado para consultas técnicas sobre redes WiFi y documentación empresarial.

## 🚀 Características Principales

### ✨ RAG Híbrido Avanzado
- **Búsqueda Vectorial**: Usando PostgreSQL con pgvector para similitud semántica
- **Búsqueda Textual**: Índice Lunr para coincidencias exactas de términos
- **Fusión RRF**: Reciprocal Rank Fusion para combinar resultados optimalmente
- **Re-ranking**: CrossEncoder para mejorar la relevancia final

### 🔧 Tecnologías Core
- **vLLM**: Generación de texto y embeddings de alta performance
- **PostgreSQL + pgvector**: Base de datos vectorial escalable
- **Streamlit**: Interfaz web interactiva
- **Marker**: Procesamiento avanzado de documentos DOCX
- **Transformers**: Modelos de embedding y tokenización

### 📄 Procesamiento de Documentos
- Soporte nativo para archivos DOCX
- Extracción inteligente de texto y tablas
- Chunking adaptativo con solapamiento
- Preservación de estructura de documentos

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documentos    │    │   Procesamiento │    │   Almacenamiento│
│     DOCX        │───▶│     Marker      │───▶│   PostgreSQL    │
└─────────────────┘    └─────────────────┘    │   + pgvector    │
                                              └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   Interfaz      │    │   RAG Híbrido   │◀──────────┘
│   Streamlit     │◀───│   Orchestrator  │
└─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │      vLLM       │
                       │   Embeddings    │
                       │  + Generación   │
                       └─────────────────┘
```

### Flujo de Búsqueda Híbrida

1. **Query del Usuario** → Procesamiento en paralelo
2. **Búsqueda Vectorial** → Embeddings + Similitud coseno
3. **Búsqueda Textual** → Índice Lunr + TF-IDF
4. **Fusión RRF** → Combina rankings con pesos optimizados
5. **Re-ranking** → CrossEncoder para refinamiento final
6. **Generación** → vLLM produce respuesta contextualizada

## 📋 Prerrequisitos

- **Python 3.8+**
- **PostgreSQL 12+** con extensión pgvector
- **vLLM Server** configurado y ejecutándose
- **8GB+ RAM** recomendado para modelos de embedding

## 🛠️ Instalación

### 1. Clonar el Repositorio
```bash
git clone <tu-repositorio>
cd rag_vllm
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias

#### Opción A: Instalación Completa (Recomendado)
```bash
pip install -r requirements.txt
```

#### Opción B: Instalación Mínima (Sin Marker)
Si prefieres una instalación más ligera sin `marker-pdf`:
```bash
pip install -r requirements-minimal.txt
```

> **Nota**: Con la instalación mínima solo podrás usar `python-docx` para procesar documentos DOCX. Marker proporciona mejor calidad de extracción pero requiere más dependencias.



### 4. Configurar Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# Configuración vLLM
VLLM_ENDPOINT=http://192.168...
VLLM_EMBED=http://192.168...
VLLM_MODEL_GENERATION=tu-modelo-generativo
EMBEDDING_API_ENDPOINT=http://192.168...

# Base de datos
DB_CONNECTION_STRING=postgresql://usuario:contraseña@localhost/rag_db
```

### 5. Crear Estructura de Directorios
```bash
mkdir -p data/docx
```

## 📚 Uso

### 1. Preparar Documentos
Coloca tus archivos DOCX en la carpeta `data/docx/`:
```
data/
└── docx/
    ├── documento1.docx
    ├── documento2.docx
    └── ...
```

### 2. Procesar e Ingestar Documentos

#### Opción A: Con Marker (Recomendado)
```bash
python scripts/ingest_data_marker.py
```

#### Opción B: Con python-docx (Alternativa)
```bash
python scripts/ingest_data.py
```


### 3. Ejecutar la Aplicación
```bash
streamlit run app/ui_main.py
```

### 4. Interactuar con el Sistema
1. Abre tu navegador en `http://localhost:8501`
2. Escribe tu pregunta sobre el contenido de los documentos
3. El sistema realizará búsqueda híbrida y generará una respuesta contextualizada

## 🔧 Configuración Avanzada

### Parámetros de Chunking
En `scripts/ingest_data_marker.py`:
```python
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer, 
    chunk_size=768,      # Tamaño del chunk
    chunk_overlap=75     # Solapamiento entre chunks
)
```

### Configuración RAG Híbrido
En `app/rag_logic.py`:
```python
# Factor para búsquedas iniciales
initial_retrieval_factor = 2

# Límite de documentos para contexto
search_limit = limit * initial_retrieval_factor

# Parámetro k para RRF
k_val = 60
```

### Modelos de Embedding
Modelos compatibles:
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
- `nomic-ai/nomic-embed-text-v1.5` (768 dim)
- Modelos personalizados vía vLLM

## 📁 Estructura del Proyecto

```
rag_vllm/
├── app/
│   ├── __init__.py
│   ├── ui_main.py              # Interfaz Streamlit
│   ├── rag_logic.py            # Lógica RAG híbrida
│   └── prompts/
│       ├── system_prompt.txt   # Prompt del sistema
│       └── wifi_expert_prompt.txt # Prompt especializado
├── scripts/
│   ├── __init__.py
│   ├── ingest_data.py          # Ingesta con python-docx
│   └── ingest_data_marker.py   # Ingesta con Marker
├── data/
│   └── docx/                   # Documentos DOCX
├── venv/                       # Entorno virtual
├── requirements.txt            # Dependencias completas
├── requirements-minimal.txt    # Dependencias sin marker-pdf
├── .env                        # Variables de entorno
├── .gitignore
└── README.md
```

### Caching
- Streamlit cachea automáticamente:
  - Documentos de la BD
  - Índices Lunr
  - Modelos CrossEncoder
  - Cliente vLLM


#### Instalación Alternativa
Si tienes problemas con `requirements.txt`:
```bash
# Instalar paso a paso las dependencias core
pip install streamlit numpy requests python-dotenv
pip install transformers sentence-transformers
pip install psycopg2-binary pgvector
pip install lunr python-docx
pip install openai
```
