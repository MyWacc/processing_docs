import os
import json
import re
from datetime import datetime
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI
import logging

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Configuración
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración para Azure services
AZURE_CONFIG = {
    "api_key": "E55fg4v8ZzwaRdgMTMcCm0D4ZEWT328USStQtzDLAaPfDaf3PjuVJQQJ99BBACYeBjFXJ3w3AAALACOGGbA9",
    "endpoint": "https://quillboot.cognitiveservices.azure.com/"
}

# Configuración de Azure OpenAI
AZURE_OPENAI_CONFIG = {
    "api_key": "BVvaEaqkEoBod3CjjYbOM9qH9qWu6VtCvwqDPHxxeY8Ygq4KHF76JQQJ99AKACYeBjFXJ3w3AAAAACOGpE1B",
    "endpoint": "https://ai-lucas8477ai303071793610.openai.azure.com/",
    "api_version": "2024-12-01-preview",
    "deployment": "o1" 
}

def extract_text_from_pdf(ruta_pdf, api_key, endpoint):
    """
    Extrae texto completo y tablas de un PDF usando Azure Document Intelligence.
    """
    logger.info(f"Extrayendo texto de PDF: {ruta_pdf}")
    
    try:
        # Crear cliente de Document Intelligence
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )
        
        # Leer archivo PDF
        with open(ruta_pdf, "rb") as f:
            pdf_content = f.read()
        
        logger.info(f"Tamaño del PDF: {len(pdf_content)} bytes")
        
        # Analizar el documento
        logger.info("Analizando documento con Azure Document Intelligence...")
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", pdf_content
        )
        result = poller.result()
        
        # Extraer texto por página
        text_by_page = []
        full_text = ""
        logger.info(f"Documento analizado. Número de páginas: {len(result.pages)}")
        
        for i, page in enumerate(result.pages):
            page_text = ""
            logger.info(f"Procesando página {i+1}: {len(page.lines)} líneas")
            for line in page.lines:
                page_text += line.content + "\n"
            text_by_page.append(page_text)
            full_text += page_text + "\n\n"
        
        # Procesar tablas
        tables_data = []
        if result.tables:
            logger.info(f"Procesando {len(result.tables)} tablas...")
            for i, table in enumerate(result.tables):
                logger.info(f"Tabla {i+1}: {table.row_count} filas x {table.column_count} columnas")
                
                # Crear matriz para esta tabla
                table_matrix = []
                for row_idx in range(table.row_count):
                    row_data = [""] * table.column_count
                    table_matrix.append(row_data)
                
                # Llenar la matriz con datos de las celdas
                for cell in table.cells:
                    if cell.row_index < len(table_matrix) and cell.column_index < len(table_matrix[0]):
                        table_matrix[cell.row_index][cell.column_index] = cell.content
                
                tables_data.append({
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "page_number": table.bounding_regions[0].page_number if table.bounding_regions else 0,
                    "cells": table_matrix
                })
        else:
            logger.info("No se encontraron tablas en el documento")
        
        return {
            "full_text": full_text,
            "text_by_page": text_by_page,
            "tables": tables_data
        }
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def convert_tables_to_text(tables_data):
    """
    Convierte las tablas extraídas a formato texto legible.
    """
    if not tables_data:
        return ""
    
    tables_text = "DATOS DE TABLAS EXTRAÍDAS:\n\n"
    
    for i, table in enumerate(tables_data):
        tables_text += f"TABLA {i+1} (Página {table['page_number']}):\n"
        
        # Determinar el ancho máximo para cada columna
        col_widths = []
        for col in range(table['column_count']):
            col_width = max([len(str(table['cells'][row][col])) for row in range(table['row_count'])] + [5])
            col_widths.append(col_width)
        
        # Crear encabezado de la tabla
        header = "+"
        for width in col_widths:
            header += "-" * (width + 2) + "+"
        
        # Imprimir cada fila
        tables_text += header + "\n"
        for row in range(table['row_count']):
            row_text = "|"
            for col in range(table['column_count']):
                cell_value = str(table['cells'][row][col])
                row_text += f" {cell_value.ljust(col_widths[col])} |"
            tables_text += row_text + "\n"
            tables_text += header + "\n"
        
        tables_text += "\n\n"
    
    return tables_text

def extract_rut_and_company(text):
    """
    Extrae el RUT y nombre de la empresa del texto.
    """
    # Buscar RUT
    rut_patterns = [
        r'RUT del emisor:\s*([0-9\-\.\s]+)',
        r'RUT\s*:\s*([0-9\-\.\s]+)',
        r'RUT\s*([0-9\-\.\s]+)',
        r'R\.?U\.?T\.?\s*([0-9\-\.\s]+)'
    ]
    
    rut = None
    for pattern in rut_patterns:
        rut_match = re.search(pattern, text, re.IGNORECASE)
        if rut_match:
            rut = rut_match.group(1).strip()
            logger.info(f"RUT encontrado: {rut}")
            break
    
    # Buscar nombre de la empresa
    name_patterns = [
        r'Nombre del emisor:\s*([^\n]+)',
        r'Razón Social\s*:\s*([^\n]+)',
        r'Nombre o Razón Social\s*([^\n]+)',
        r'RUT.*\s+([A-Za-z\s]+SPA)',
        r'RUT.*\s+([A-Za-z\s]+S\.A\.)'
    ]
    
    company_name = None
    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.IGNORECASE)
        if name_match:
            company_name = name_match.group(1).strip()
            logger.info(f"Nombre de empresa encontrado: {company_name}")
            break
    
    return rut, company_name

def extract_financial_years(text):
    """
    Identifica los años fiscales disponibles en el documento.
    """
    logger.info("Extrayendo años fiscales del documento...")
    
    # Buscar años en declaraciones de renta
    f22_years = re.findall(r'Año Tributario (\d{4})', text)
    logger.info(f"Años encontrados en declaraciones de renta (F22): {f22_years}")
    
    # Buscar años en declaraciones mensuales
    f29_years = set(re.findall(r'PERIODO 15 \d{2} / (\d{4})', text))
    logger.info(f"Años encontrados en declaraciones mensuales (F29): {f29_years}")
    
    # Buscar años en otros formatos
    other_years = set(re.findall(r'20\d{2}-12-31', text))
    other_years = {year[:4] for year in other_years}
    logger.info(f"Años encontrados en otros formatos: {other_years}")
    
    # Combinar y ordenar años
    all_years = sorted(list(set(f22_years) | f29_years | other_years), reverse=True)
    logger.info(f"Todos los años identificados: {all_years}")
    
    return all_years

def extract_data_with_openai(pdf_text, tables_text, openai_config, empresa, industria):
    """
    Usa Azure OpenAI para extraer datos financieros a partir del texto y tablas del PDF.
    """
    logger.info("Procesando datos con Azure OpenAI...")
    
    try:
        # Crear cliente de Azure OpenAI
        client = AzureOpenAI(
            api_version=openai_config["api_version"],
            azure_endpoint=openai_config["endpoint"],
            api_key=openai_config["api_key"]
        )
        
        # Obtener información básica de la empresa del texto
        rut, nombre_empresa = extract_rut_and_company(pdf_text)
        if not empresa and nombre_empresa:
            empresa = nombre_empresa
        
        # Identificar años fiscales
        años_identificados = extract_financial_years(pdf_text)
        if not años_identificados:
            años_identificados = ["2023", "2022", "2021"]
        
        # Crear prompt para extracción financiera
        prompt = f"""
        Eres un experto contador y analista financiero especializado en datos tributarios chilenos. Tu tarea es analizar detalladamente esta carpeta tributaria de la empresa {empresa} y extraer todos los datos financieros relevantes.

        INSTRUCCIONES IMPORTANTES:
        1. Analiza cuidadosamente todo el texto de la carpeta tributaria, buscando todas las declaraciones F29 y F22 para los años disponibles ({', '.join(años_identificados)}).
        2. Para los ingresos, suma todas las "BASE IMPONIBLE" mensuales de las declaraciones F29 para cada año.
        3. Extrae datos del balance y estado de resultados, prestando especial atención a:
           - Total Activos (TotalActivos)
           - Activos Circulantes (TACirculante)
           - Activos Fijos (TAFijo)
           - Pasivos Circulantes (TPCirculantes)
           - Pasivos Largo Plazo (TPLPlazo)
           - Pasivos Exigibles Total (TPExigible)
           - Capital Enterado (CEnterado)
           - Otras Partidas No Exigibles (OPNExigibles)
           - Utilidad del Ejercicio (UEjercicio)
           - Caja, Banco, Existencias, Otros Circulantes, Provisiones
        4. Para el estado de resultados, extrae:
           - Ingresos
           - Costos Directos
           - Depreciación
           - Margen Bruto
           - Otros Gastos
           - Utilidad Neta
        5. Para valores que no encuentres explícitamente, usa 0 en lugar de estimaciones.
        
        DATOS DE LA CARPETA TRIBUTARIA:

        {pdf_text}

        {tables_text}

        Basado en toda esta información, genera un objeto JSON con la siguiente estructura exacta:

        ```json
        {{
          "Respuestas": {{
            "AnalisisTributario": {{
              "AnalisisXML": {{
                "Analisis": {{
                  "Balance": {{
                    "Periodo": [
                      {{
                        "Anio": "{años_identificados[0] if len(años_identificados) > 0 else '2023'}",
                        "TotalActivos": 0,
                        "TACirculante": 0,
                        "TAFijo": 0,
                        "TPCirculantes": 0,
                        "TPLPlazo": 0,
                        "TPExigible": 0,
                        "CEnterado": 0,
                        "OPNExigibles": 0,
                        "UEjercicio": 0,
                        "Caja": 0,
                        "Banco": 0,
                        "Existencia": 0,
                        "OCirculantes": 0,
                        "Provisiones": 0
                      }},
                      {{
                        "Anio": "{años_identificados[1] if len(años_identificados) > 1 else '2022'}",
                        "TotalActivos": 0,
                        "TACirculante": 0,
                        "TAFijo": 0,
                        "TPCirculantes": 0,
                        "TPLPlazo": 0,
                        "TPExigible": 0,
                        "CEnterado": 0,
                        "OPNExigibles": 0,
                        "UEjercicio": 0,
                        "Caja": 0,
                        "Banco": 0,
                        "Existencia": 0,
                        "OCirculantes": 0,
                        "Provisiones": 0
                      }},
                      {{
                        "Anio": "{años_identificados[2] if len(años_identificados) > 2 else '2021'}",
                        "TotalActivos": 0,
                        "TACirculante": 0,
                        "TAFijo": 0,
                        "TPCirculantes": 0,
                        "TPLPlazo": 0,
                        "TPExigible": 0,
                        "CEnterado": 0,
                        "OPNExigibles": 0,
                        "UEjercicio": 0,
                        "Caja": 0,
                        "Banco": 0,
                        "Existencia": 0,
                        "OCirculantes": 0,
                        "Provisiones": 0
                      }}
                    ]
                  }},
                  "EERR1": {{
                    "Periodo": [
                      {{
                        "Anio": "{años_identificados[0] if len(años_identificados) > 0 else '2023'}",
                        "Ingresos": {{"#text": "0"}},
                        "Costos": {{"#text": "0"}},
                        "Depreciacion": {{"#text": "0"}},
                        "Margen": {{"#text": "0"}},
                        "OGastos": {{"#text": "0"}},
                        "UNeta": {{"#text": "0"}}
                      }},
                      {{
                        "Anio": "{años_identificados[1] if len(años_identificados) > 1 else '2022'}",
                        "Ingresos": {{"#text": "0"}},
                        "Costos": {{"#text": "0"}},
                        "Depreciacion": {{"#text": "0"}},
                        "Margen": {{"#text": "0"}},
                        "OGastos": {{"#text": "0"}},
                        "UNeta": {{"#text": "0"}}
                      }},
                      {{
                        "Anio": "{años_identificados[2] if len(años_identificados) > 2 else '2021'}",
                        "Ingresos": {{"#text": "0"}},
                        "Costos": {{"#text": "0"}},
                        "Depreciacion": {{"#text": "0"}},
                        "Margen": {{"#text": "0"}},
                        "OGastos": {{"#text": "0"}},
                        "UNeta": {{"#text": "0"}}
                      }}
                    ]
                  }}
                }}
              }}
            }}
          }},
          "RUT": "{rut if rut else 'RUT_DESCONOCIDO'}",
          "RazonSocial": "{empresa if empresa else 'Empresa_Desconocida'}"
        }}
        ```

        Reemplaza los valores "0" con los datos numéricos reales que encuentres en el documento. Responde SOLAMENTE con el JSON, sin texto explicativo adicional.
        """
        
        # Llamar a la API
        logger.info("Enviando solicitud a Azure OpenAI...")
        response = client.chat.completions.create(
            model=openai_config["deployment"],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200000
        )
        
        # Extraer respuesta
        response_text = response.choices[0].message.content
        logger.info(f"Respuesta recibida de Azure OpenAI ({len(response_text)} caracteres)")
        
        # Guardar respuesta para diagnóstico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = f"{OUTPUT_DIR}/openai_response_{timestamp}.txt"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response_text)
        logger.info(f"Respuesta guardada en {response_file}")
        
        # Extraer JSON de la respuesta
        json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text)
        
        if json_match:
            json_str = json_match.group(1)
            logger.info("JSON extraído de la respuesta")
        else:
            # Intentar encontrar el JSON sin los delimitadores de código
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                logger.info("JSON extraído sin delimitadores")
            else:
                logger.error("No se pudo encontrar JSON válido en la respuesta")
                return None
        
        try:
            # Cargar el JSON
            data = json.loads(json_str)
            logger.info("JSON parseado exitosamente")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar JSON: {e}")
            logger.error(f"JSON problemático: {json_str[:500]}...")
            return None
            
    except Exception as e:
        logger.error(f"Error en extract_data_with_openai: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_pdf(ruta_pdf, empresa=None, industria="Electrical Equipment"):
    """
    Procesa un archivo PDF con Azure Document Intelligence y Azure OpenAI.
    """
    logger.info(f"Procesando PDF: {ruta_pdf}")
    
    # Paso 1: Extraer texto y tablas del PDF
    pdf_data = extract_text_from_pdf(ruta_pdf, AZURE_CONFIG["api_key"], AZURE_CONFIG["endpoint"])
    
    if not pdf_data:
        logger.error("Error al extraer texto del PDF")
        return None
    
    # Paso 2: Guardar texto completo para diagnóstico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    texto_completo_path = f"{OUTPUT_DIR}/texto_completo_{timestamp}.txt"
    with open(texto_completo_path, "w", encoding="utf-8") as f:
        f.write(pdf_data["full_text"])
    logger.info(f"Texto completo guardado en {texto_completo_path}")
    
    # Paso 3: Convertir tablas a formato texto para incluir en el prompt
    tables_text = convert_tables_to_text(pdf_data["tables"])
    
    # Paso 4: Detectar automáticamente la empresa si no se proporcionó
    if not empresa:
        rut, nombre_empresa = extract_rut_and_company(pdf_data["full_text"])
        if nombre_empresa:
            empresa = nombre_empresa
            logger.info(f"Empresa detectada automáticamente: {empresa}")
        else:
            empresa = "Empresa_Desconocida"
            logger.warning("No se pudo detectar el nombre de la empresa. Usando nombre por defecto.")
    
    # Paso 5: Extraer datos financieros con Azure OpenAI
    datos_extraidos = extract_data_with_openai(
        pdf_data["full_text"], 
        tables_text, 
        AZURE_OPENAI_CONFIG, 
        empresa, 
        industria
    )
    
    if datos_extraidos:
        # Guardar los datos extraídos
        output_file = f"{OUTPUT_DIR}/{empresa}_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(datos_extraidos, f, indent=2)
        logger.info(f"Datos extraídos guardados en {output_file}")
    else:
        logger.error("No se pudieron extraer datos del PDF")
    
    return datos_extraidos

if __name__ == "__main__":
    import sys
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        logger.error("Uso: python pdf_financial_extractor.py ruta/al/archivo.pdf [nombre_empresa] [industria]")
        sys.exit(1)
    
    # Obtener ruta del archivo
    ruta_pdf = sys.argv[1]
    
    # Obtener nombre de la empresa (opcional)
    empresa = None
    if len(sys.argv) > 2 and sys.argv[2]:
        empresa = sys.argv[2]
    
    # Obtener industria (opcional)
    industria = "Electrical Equipment"  # Valor por defecto
    if len(sys.argv) > 3 and sys.argv[3]:
        industria = sys.argv[3]
    
    # Procesar PDF
    resultado = process_pdf(ruta_pdf, empresa, industria)
    
    if resultado:
        logger.info("Procesamiento completado exitosamente.")
        sys.exit(0)
    else:
        logger.error("Error en el procesamiento del PDF.")
        sys.exit(1)