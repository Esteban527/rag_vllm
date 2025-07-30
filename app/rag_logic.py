import psycopg2 
from pgvector.psycopg2 import register_vector 
import numpy as np 
import requests

# --- Funciones de Lógica RAG ---

def get_all_documents(db_connection_string: str) -> list:
    """
    Recupera todos los documentos (id, texto, embedding) de PostgreSQL.
    """
    all_docs = []
    try:
        conn = psycopg2.connect(db_connection_string)
        register_vector(conn) # Habilita el manejo de pgvector.
        with conn.cursor() as cur:
            cur.execute("SELECT id, text_content, embedding FROM documents_security")
            rows = cur.fetchall()
            for row_data in rows:
                embedding_val = row_data[2]
                embedding_array = np.array(embedding_val) if embedding_val is not None else None
                all_docs.append({"id": row_data[0], "text": row_data[1], "embedding": embedding_array})
        conn.close()
    except Exception as e:
        print(f"Error en get_all_documents: {e}")
    return all_docs

def full_text_search(query: str, limit: int, lunr_search_index, documents_by_id_map: dict) -> list:
    """Realiza búsqueda de texto completo usando un índice Lunr pre-construido."""
    if not lunr_search_index:
        print("Error: Índice Lunr no proporcionado para full_text_search.")
        return []
    try:
        results = lunr_search_index.search(query)
        # Mapea los IDs de los resultados de Lunr a los documentos completos.
        return [documents_by_id_map[result["ref"]] for result in results[:limit] if result["ref"] in documents_by_id_map]
    except Exception as e:
        print(f"Error en full_text_search: {e}")
        return []

def vector_search(query: str, limit: int, db_conn_str: str, vllm_embed: str) -> list:
    """Realiza búsqueda vectorial directamente en la base de datos PostgreSQL usando pgvector."""
    found_docs_list = []
    if not vllm_embed:
        print("Error: Endpoint vLLM no funciona para vector_search.")
        return []
    try:
        # Genera el embedding para la consulta del usuario usando Ollama API.
        embed_url = f"{vllm_embed}"
        payload = {
            "text": query
        }
        print("foo")
        response = requests.post(embed_url, json=payload, timeout=60)
        print("bar")
        print(response)
        response.raise_for_status()
        print("baz")
        query_embed_data = response.json()["embedding"]
        query_embed_np = np.array(query_embed_data)

        # Conecta a la BD y realiza la búsqueda de similitud vectorial.
        conn = psycopg2.connect(db_conn_str)
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id, text_content, embedding FROM documents_security ORDER BY embedding <=> %s DESC LIMIT %s", 
                        (query_embed_np, limit))
            rows_data = cur.fetchall()
            for row_item in rows_data:
                embed_val = row_item[2]
                embed_arr = np.array(embed_val) if embed_val is not None else None
                found_docs_list.append({"id": row_item[0], "text": row_item[1], "embedding": embed_arr})
        conn.close()
    except Exception as e:
        print(f"Error en vector_search: {e}")
    return found_docs_list

def reciprocal_rank_fusion(text_results_list: list, vector_results_list: list, documents_by_id_map: dict, k_val: int = 60) -> list:
    """Combina los resultados de la búsqueda de texto y vectorial usando Reciprocal Rank Fusion (RRF)."""
    scores_map = {} # Para almacenar las puntuaciones RRF de cada documento.
    
    # Filtra y procesa resultados de la búsqueda de texto.
    valid_text_results = [doc for doc in text_results_list if doc["id"] in documents_by_id_map]
    for i, doc_item in enumerate(valid_text_results):
        doc_id_val = doc_item["id"]
        scores_map.setdefault(doc_id_val, 0)
        scores_map[doc_id_val] += 1 / (i + k_val) # Fórmula RRF.
    
    # Filtra y procesa resultados de la búsqueda vectorial.
    valid_vector_results = [doc for doc in vector_results_list if doc["id"] in documents_by_id_map]
    for i, doc_item in enumerate(valid_vector_results):
        doc_id_val = doc_item["id"]
        scores_map.setdefault(doc_id_val, 0)
        scores_map[doc_id_val] += 1 / (i + k_val)
    
    if not scores_map: 
        print("No se encontraron documentos válidos para fusionar.")
        return [] 
    
    # Ordena los documentos por su puntuación RRF total.
    ranked_doc_ids = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
    # Devuelve los documentos completos, ordenados.
    return [documents_by_id_map[doc_id_val] for doc_id_val, _ in ranked_doc_ids if doc_id_val in documents_by_id_map]

def rerank(query: str, retrieved_docs_list: list, cross_encoder_model_inst) -> list:
    """Reclasifica una lista de documentos recuperados usando un modelo CrossEncoder para mejorar la relevancia."""
    if not cross_encoder_model_inst or not retrieved_docs_list:
        return retrieved_docs_list # Devuelve original si no hay modelo o docs.
    
    # Prepara los pares (consulta, texto_documento) para el CrossEncoder.
    sentence_pairs_list = [(query, doc_item["text"]) for doc_item in retrieved_docs_list if doc_item.get("text")]
    if not sentence_pairs_list: return retrieved_docs_list # Si no hay textos válidos.

    try:
        # Obtiene las puntuaciones de relevancia del CrossEncoder.
        cross_scores = cross_encoder_model_inst.predict(sentence_pairs_list)
        # Empareja los documentos con sus nuevas puntuaciones y los reordena.
        docs_with_text = [doc_item for doc_item in retrieved_docs_list if doc_item.get("text")]
        reranked_docs = [doc_item for _, doc_item in sorted(zip(cross_scores, docs_with_text), key=lambda x: x[0], reverse=True)]
        return reranked_docs
    except Exception as e:
        print(f"Error en rerank: {e}")
        return retrieved_docs_list # Devuelve original en caso de error.

def hybrid_search_orchestrator(
    query: str, 
    limit: int,  # Límite final de documentos a devolver.
    lunr_search_idx,  # Índice Lunr pre-cargado.
    docs_by_id_map: dict, # Mapa de ID -> documento (para Lunr y RRF).
    embeddings_url,  # Cliente Ollama para generar embedding de la consulta.
    cross_enc_model_inst, # Modelo CrossEncoder para re-ranking.
    db_conn_str_param: str, # Cadena de conexión a la BD para la búsqueda vectorial.
    ) -> list:
    """
    Orquesta el pipeline de búsqueda híbrida: texto + vectorial, fusión RRF, y re-ranking.
    """

    initial_retrieval_factor = 2 # Factor para obtener más resultados en las búsquedas iniciales.
    search_limit = limit * initial_retrieval_factor # Límite para las búsquedas individuales (texto y vector).

    # Búsqueda de texto completo (Lunr).
    text_search_results = full_text_search(query, search_limit, lunr_search_idx, docs_by_id_map)

    # Búsqueda vectorial (directamente en la base de datos pgvector).
    vector_search_results = vector_search(query, search_limit, db_conn_str_param, embeddings_url)
        
    # Fusión de los resultados de ambas búsquedas usando RRF.
    fused_search_results = reciprocal_rank_fusion(text_search_results, vector_search_results, docs_by_id_map)

    # Re-ranking de los resultados fusionados con CrossEncoder.
    reranked_search_results = rerank(query, fused_search_results, cross_enc_model_inst)
    
    # Devuelve el número deseado ('limit') de los documentos mejor clasificados.
    return reranked_search_results[:limit]