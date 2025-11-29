import pandas as pd
import re

def convertir_fecha(fecha_escrita):
    # Recibe una fecha en formato "12 de marzo de 2023" y la convierte a "12/03/2023"

    meses = {
    'enero': '01',
    'febrero': '02',
    'marzo': '03',
    'abril': '04',
    'mayo': '05',
    'junio': '06',
    'julio': '07',
    'agosto': '08',
    'septiembre': '09',
    'octubre': '10',
    'noviembre': '11',
    'diciembre': '12'
    }
    
    partes = fecha_escrita.split(' de ')
    dia = partes[0].zfill(2)  # Asegura 2 dígitos para días 1-9
    mes = meses[partes[1]]
    año = partes[2]
    return f"{dia}/{mes}/{año}"  # Formato DD/MM/AAAA

def aplicar_funcion_fecha(corpus_completo):
    # Recibe un DataFrame con una columna 'Fecha' en formato "12 de marzo de 2023"
    # Devuelve el DataFrame con la columna 'Fecha' convertida a formato datetime
    corpus_completo['Fecha'] = corpus_completo['Fecha'].apply(convertir_fecha)
    corpus_completo['Fecha'] = pd.to_datetime(corpus_completo['Fecha'], format='%d/%m/%Y', errors='coerce')
    return corpus_completo


def limpiar_texto(texto):
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    # Eliminar menciones y hashtags
    texto = re.sub(r'\@\w+|\#','', texto) 
    # Eliminar caracteres no deseados, excepto puntuación común
    texto = re.sub(r'[^0-9A-Za-záéíóúÁÉÍÓÚñÑüÜ\s\.\,\;\:\!\?\¿\¡]', '', texto)
    # Convertir a minúsculas
    # texto = texto.lower()
    # Eliminar espacios extra
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def aplicar_funcion_limpieza(corpus):
    corpus['Texto_limpio'] = corpus['Texto'].apply(limpiar_texto)
    return corpus

def dividir_en_chunks_solapados(texto, tamaño=512, solapamiento=50, umbral_minimo=50):
    """
    Divide un texto en chunks de 'tamaño' palabras con solapamiento,
    asegurando NO repetir el último segmento.
    """

    palabras = texto.split()
    longitud = len(palabras)

    if longitud == 0:
        return []

    chunks = []
    paso = tamaño - solapamiento
    i = 0

    while i < longitud:
        # Crear chunk
        chunk = palabras[i: i + tamaño]

        # Agregar si no está vacío
        if chunk:
            chunks.append(" ".join(chunk))

        # Avanzar al siguiente inicio
        i += paso

        # Si el siguiente inicio ya está dentro del último chunk, romper
        if i >= longitud:
            break

    # --- Ajustar el último chunk si es demasiado corto ---
    if len(chunks) > 1:
        ultimo = chunks[-1].split()
        if len(ultimo) < umbral_minimo:
            # fusionar sin duplicaciones:
            anteultimo = chunks[-2].split()

            # palabras que no se repiten
            diferencia = ultimo[len(anteultimo):] if len(ultimo) > len(anteultimo) else ultimo

            # si diferencia es prácticamente nada, mejor descartar
            if len(diferencia) == 0:
                chunks.pop(-1)
            else:
                chunks[-2] = " ".join(anteultimo + diferencia)  # unir sin repetir
                chunks.pop(-1)

    return chunks


def crear_chunks(corpus, columna_texto="Texto_limpio", tamaño=512, solapamiento=50, umbral_minimo=50):
    """
    Recibe un DataFrame con una columna de texto.
    Retorna un nuevo DataFrame con los textos divididos en chunks.
    """
    data = []
    for idx, fila in corpus.iterrows():
        chunks = dividir_en_chunks_solapados(fila[columna_texto], tamaño, solapamiento, umbral_minimo)
        for num_chunk, chunk in enumerate(chunks):
            data.append({
                "id_doc": idx+1,
                "autor_doc": fila.get("Autor", None),
                "fecha_doc": fila.get("Fecha", None),
                "diario_doc": fila.get("Diario", None),
                "titulo_doc": fila.get("Título", None),
                "chunk_id": num_chunk,
                "texto_chunk": chunk
            })
    return pd.DataFrame(data)