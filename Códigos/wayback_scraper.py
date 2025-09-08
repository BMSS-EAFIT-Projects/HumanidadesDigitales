import requests
import pandas as pd

def get_wayback_urls(domain, from_year, to_year, output_file="wayback_urls.xlsx"):
    """
    Consulta la API CDX de Wayback Machine para obtener todas las URLs archivadas de un dominio.
    
    Args:
        domain (str): dominio o patrón de URLs, ej: "elespectador.com/opinion/*"
        from_year (int): año inicial del rango
        to_year (int): año final del rango
        output_file (str): nombre del archivo Excel de salida
    """

    # Endpoint CDX API
    cdx_url = "http://web.archive.org/cdx/search/cdx"

    # Parámetros de la consulta
    params = {
        "url": domain,
        "from": from_year,
        "to": to_year,
        "output": "json",
        "fl": "timestamp,original",
        "filter": "statuscode:200",
        "collapse": "digest"
    }

    print(f"🔎 Consultando CDX API para {domain} entre {from_year}-{to_year}...")

    # Hacer la petición
    response = requests.get(cdx_url, params=params)

    if response.status_code != 200:
        print("❌ Error en la consulta:", response.status_code)
        return None

    data = response.json()

    # La primera fila son los encabezados
    headers = data[0]
    rows = data[1:]

    # Convertir a DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Crear columna con URL archivada
    df["archived_url"] = df["timestamp"].apply(
        lambda ts: f"http://web.archive.org/web/{ts}/{df.loc[df['timestamp']==ts, 'original'].values[0]}"
    )

    # Guardar en Excel
    df.to_excel(output_file, index=False)
    print(f"✅ Archivo guardado como {output_file} con {len(df)} registros.")

    return df


# Ejemplo de uso:
# Todas las columnas de opinión de El Espectador entre 2018 y 2020
df = get_wayback_urls("elespectador.com/opinion/*", 2018, 2019)
