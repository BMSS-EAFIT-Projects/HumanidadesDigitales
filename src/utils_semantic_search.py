import pandas as pd
import glob
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
import sys
import os
import matplotlib.pyplot as plt


# PARTE 1: Cargar términos y crear embeddings por subcategoría

def cargar_subcategorias(path_dir, model):
    """
    Lee los archivos .txt en el directorio.
    En lugar de promediar embeddings de cada término, concatena
    todas las palabras en una sola frase y obtiene un único embedding.
    """
    subcat_embeddings = {}

    for file_path in glob.glob(os.path.join(path_dir, "*.txt")):
        subcat = os.path.splitext(os.path.basename(file_path))[0]

        # Leer términos
        with open(file_path, "r", encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip()]
        
        # Concatenar todo en una sola frase
        large_phrase = " ".join(terms)

        # Generar embedding único
        embedding = model.encode(large_phrase, 
                                 convert_to_tensor=True, 
                                 show_progress_bar=False,
                                prompt_name="STS")

        subcat_embeddings[subcat] = embedding
    
    return subcat_embeddings


# PARTE 2: Calcular embeddings chunks

def obtener_embeddings_chunks(
    chunks_df,
    model,
    batch_size=64,
    save_path="../data/processed/chunk_embeddings.npy",
    RELOAD=False
):
    """
    Calcula o carga embeddings de los chunks según el valor de RELOAD.
    
    Args:
        chunks_df (pd.DataFrame): DataFrame con columna 'texto_chunk'
        model: modelo de sentence-transformers
        batch_size (int): tamaño del lote para encoding
        save_path (str): ruta donde guardar/cargar embeddings
        RELOAD (bool): si True, recalcula embeddings aunque exista archivo
    
    Returns:
        np.ndarray con embeddings de los chunks
    """
    
    if not RELOAD and os.path.exists(save_path):
        print(f"Embeddings encontrados en {save_path}, cargando...")
        embeddings = np.load(save_path)
        print(f"Embeddings cargados con forma: {embeddings.shape}")
        return embeddings
    
    print("Calculando embeddings desde cero...")
    textos = chunks_df["texto_chunk"].tolist()
    chunk_embeddings = model.encode(
        textos,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    embeddings_np = chunk_embeddings.cpu().numpy()

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Guardar embeddings
    np.save(save_path, embeddings_np)
    print(f"Embeddings calculados y guardados en {save_path}")

    return embeddings_np

# PARTE 3: Calcular similitudes

def calcular_similitudes_chunks(chunks_df, chunk_embeddings, subcat_embeddings):
    """
    Calcula similitudes coseno entre embeddings de chunks y de subcategorías.
    
    Args:
        chunks_df (pd.DataFrame): DataFrame original con 'chunk_id', 'id_doc', 'texto_chunk'
        chunk_embeddings (np.ndarray o tensor): embeddings precomputados
        subcat_embeddings (dict): {subcat: tensor de embedding promedio}
    
    Returns:
        pd.DataFrame con similitudes por subcategoría
    """
    # Convertir a tensores
    chunk_embeddings = torch.tensor(chunk_embeddings)
    subcats = list(subcat_embeddings.keys())
    subcat_matrix = torch.stack(list(subcat_embeddings.values()))

    # Calcular similitudes en bloque
    sim_matrix = util.cos_sim(chunk_embeddings, subcat_matrix).cpu().numpy()

    # Armar DataFrame de resultados
    sim_df = pd.DataFrame(sim_matrix, columns=subcats)
    sim_df.insert(0, "chunk_id", chunks_df["chunk_id"].values)
    sim_df.insert(1, "id_doc", chunks_df["id_doc"].values)
    sim_df.insert(2, "texto_chunk", chunks_df["texto_chunk"].values)

    return sim_df


# PARTE 4: Aplicar umbral y asignar categorías

def asignar_categorias(df, umbral=0.30):
    """
    Añade columnas con las categorías detectadas por chunk
    y sus scores.
    """
    categorias_detectadas = []
    for _, fila in df.iterrows():
        cats = []
        for col in df.columns:
            if col not in ["chunk_id", "id_doc", "texto_chunk"]:
                if fila[col] >= umbral:
                    cats.append((col, fila[col]))
        categorias_detectadas.append(cats if cats else [("ninguna", 0)])
    
    df["categorias_detectadas"] = categorias_detectadas
    return df

def calcular_similitudes_documentos(chunks_df, chunk_embeddings, subcat_embeddings):
    """
    Calcula similitudes por documento (id_doc) promediando los embeddings de sus chunks.
    
    Args:
        chunks_df (pd.DataFrame): contiene columnas [chunk_id, id_doc, texto_chunk]
        chunk_embeddings (np.ndarray): embeddings por chunk (en mismo orden que chunks_df)
        subcat_embeddings (dict): {subcat: embedding_tensor}
    
    Returns:
        pd.DataFrame con una fila por documento y similidades por subcategoría.
    """
    # Convertir a tensor
    chunk_emb_tensor = torch.tensor(chunk_embeddings)

    # Agrupar chunks por id_doc
    docs = chunks_df["id_doc"].unique()

    doc_embeddings = []
    doc_ids = []

    for doc in docs:
        idxs = chunks_df.index[chunks_df["id_doc"] == doc].tolist()
        emb = chunk_emb_tensor[idxs]          # embeddings de los chunks del doc
        emb_mean = emb.mean(dim=0)            # PROMEDIO
        doc_embeddings.append(emb_mean)
        doc_ids.append(doc)

    # Matriz final
    doc_emb_tensor = torch.stack(doc_embeddings)

    # Subcategorías
    subcats = list(subcat_embeddings.keys())
    subcat_matrix = torch.stack(list(subcat_embeddings.values()))

    # Similitud coseno
    sim_matrix = util.cos_sim(doc_emb_tensor, subcat_matrix).cpu().numpy()

    # DataFrame resultado
    sim_doc_df = pd.DataFrame(sim_matrix, columns=subcats)
    sim_doc_df.insert(0, "id_doc", doc_ids)
    #sim_doc_df.insert(1, "texto_chunk", chunks_df["texto_chunk"].values)


    return sim_doc_df
