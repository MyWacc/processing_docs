import os
import json
import logging
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Azure / OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURACIONES
# ============================================================================
# Ajusta con tus credenciales reales y endpoints
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
MAX_WORKERS = 5  # Número de hilos para procesar páginas en paralelo

# ============================================================================
# 1) EXTRACCIÓN DE TEXTO (OCR) MEDIANTE AZURE DOCUMENT INTELLIGENCE
# ============================================================================
def extract_text_from_pdf(ruta_pdf, api_key, endpoint):
    """
    Extrae texto y tablas de un PDF usando Azure Document Intelligence.
    Devuelve un dict con 'full_text', 'text_by_page', 'tables'.
    """
    logger.info(f"Extrayendo texto de PDF: {ruta_pdf}")
    try:
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        with open(ruta_pdf, "rb") as f:
            pdf_content = f.read()

        logger.info(f"Tamaño del PDF: {len(pdf_content)} bytes")
        poller = client.begin_analyze_document("prebuilt-layout", pdf_content)
        result = poller.result()
        logger.info(f"Documento analizado. Número de páginas: {len(result.pages)}")

        # Extraer texto por página
        text_by_page = []
        full_text = ""
        for i, page in enumerate(result.pages):
            page_text = ""
            for line in page.lines:
                page_text += line.content + "\n"
            text_by_page.append(page_text)
            full_text += page_text + "\n\n"

        # Extraer tablas (opcional)
        tables_data = []
        if result.tables:
            logger.info(f"Procesando {len(result.tables)} tablas...")
            for i, table in enumerate(result.tables):
                table_matrix = []
                for row_idx in range(table.row_count):
                    row_data = [""] * table.column_count
                    table_matrix.append(row_data)
                for cell in table.cells:
                    if (cell.row_index < len(table_matrix) and
                        cell.column_index < len(table_matrix[0])):
                        table_matrix[cell.row_index][cell.column_index] = cell.content
                page_number = table.bounding_regions[0].page_number if table.bounding_regions else 0
                tables_data.append({
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "page_number": page_number,
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
        logger.error(f"Error al extraer texto del PDF: {e}")
        return None

# ============================================================================
# 2) FUNCIÓN PARA PROCESAR 1 PÁGINA (LLAMADA A GPT)
# ============================================================================
def extract_data_from_page(page_index, page_text, openai_client):
    """
    Llama a GPT para extraer datos de UNA página.
    Devuelve un dict con la info parcial que GPT logre encontrar.
    Guardamos también un archivo JSON parcial por si quieres diagnosticar.
    """
    prompt = f"""
Eres un experto financiero y tributario. Aquí tienes el texto completo de la página {page_index}:

--- INICIO DE LA PÁGINA ---
{page_text}
--- FIN DE LA PÁGINA ---

Extrae información relevante de F29, F22, balances, etc., 
y devuélvela en un JSON válido. Sin explicaciones adicionales.
Si algún dato no está, usa 0.

Ejemplo de JSON parcial:
{{
  "f29_partial": {{
    "BaseImponible": 123456,
    "RetImpUnicoTrab": 1234
  }},
  "f22_partial": {{
    "AñoTributario": 2024,
    "RentaLiquidaImponible": 0
  }}
}}
""".strip()

    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CONFIG["deployment"],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=50000,  # Ajusta según tu chunk/página
        )
        content = response.choices[0].message.content

        # Buscar JSON en la respuesta
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            try:
                partial_data = json.loads(json_str)
            except json.JSONDecodeError as je:
                logger.warning(f"JSON malformado en página {page_index}, error: {je}")
                partial_data = {}
        else:
            logger.warning(f"No se encontró JSON claro en la respuesta del modelo para página {page_index}.")
            partial_data = {}
        
        # Guardar el JSON parcial de diagnóstico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partial_file = f"{OUTPUT_DIR}/partial_page_{page_index}_{timestamp}.json"
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump(partial_data, f, indent=2)
        logger.info(f"Datos parciales guardados en {partial_file}")
        
        return partial_data

    except Exception as e:
        logger.error(f"Error con OpenAI en página {page_index}: {e}")
        return {}

# ============================================================================
# 3) COMBINAR LOS RESULTADOS PARCIALES
# ============================================================================
def combine_partial_results(partial_results_list):
    """
    Une la información parcial de cada página en una estructura final
    (igual a tu JSON de ejemplo). Para que no quede todo en 0, 
    DEBES sumar o asignar lo que devuelva GPT en los partial_data.
    """
    # -- Plantilla final --
    final_data = {
        "quantitative_input": {
            "statement_income": {
                "first_year": {
                    "date": "2024-12-31",
                    "net_revenue": 0.0,
                    "cogs": 0.0,
                    "ebit_ajusted": 0.0,
                    "interest_income": 0,
                    "financial_expenses": 0,
                    "effective_tax": 0.0,
                    "ebitda": 0.0,
                    "gross_profit": 0.0,
                    "sg_a": 0.0,
                    "other_income_expenses": 0.0,
                    "depreciacion": 0.0,
                    "net_income": 0.0,
                    "compra_de_activos_fijos": 0,
                    "d_a": 0,
                    "gain_loss_difference_exchange_rates": 0,
                    "amortizacion": 0,
                    "compra_de_intangibles": 0
                },
                "second_year": {
                    "date": "2023-12-31",
                    "net_revenue": 0.0,
                    "cogs": 0.0,
                    "ebit_ajusted": 0.0,
                    "interest_income": 0,
                    "financial_expenses": 0,
                    "effective_tax": 0.0,
                    "ebitda": 0.0,
                    "gross_profit": 0.0,
                    "sg_a": 0.0,
                    "other_income_expenses": 0.0,
                    "depreciacion": 0.0,
                    "net_income": 0.0,
                    "compra_de_activos_fijos": 0,
                    "d_a": 0,
                    "gain_loss_difference_exchange_rates": 0,
                    "amortizacion": 0,
                    "compra_de_intangibles": 0
                },
                "third_year": {
                    "date": "2022-12-31",
                    "net_revenue": 0.0,
                    "cogs": 0.0,
                    "ebit_ajusted": 0.0,
                    "interest_income": 0,
                    "financial_expenses": 0,
                    "effective_tax": 0.0,
                    "ebitda": 0.0,
                    "gross_profit": 0.0,
                    "sg_a": 0.0,
                    "other_income_expenses": 0.0,
                    "depreciacion": 0.0,
                    "net_income": 0.0,
                    "compra_de_activos_fijos": 0,
                    "d_a": 0,
                    "gain_loss_difference_exchange_rates": 0,
                    "amortizacion": 0,
                    "compra_de_intangibles": 0
                }
            },
            "parameter_inputs": {
                "currency": "CLP",
                "industry": "Electrical Equipment",
                "country": "Chile",
                "region": "Emerging Markets",
                "company": "MMYY",
                "units": "-",
                "last_periods": "2024-12-31"
            },
            "data_continuation": {
                "high_growth_period": 2,
                "caja_operativa_ventas": 0.02,
                "politica_de_dividendos": 0,
                "tasa_de_impuesto_marginal": 0.27,
                "calculo_kd_con_synthetic_spread": "Si",
                "calculo_de_kd_con_coverage_ratio_historico": "Si",
                "caja_minima": 0.01,
                "perdida_de_arrastre_inicial_en_miles": None,
                "tasa_nueva_deuda_anual": 0.07,
                "plazo_nueva_deuda_anos": 6,
                "monto_nueva_deuda_en_miles": 0,
                "revenue_cagr": None,
                "margen": -40.83,
                "margen_ebit": None,
                "tasa_estatuitaria_local": 0.27,
                "depreciación_ventas": None,
                "non_cash_wc_sales": None,
                "sales_invested_capital": None,
                "crecimiento_en_perpetuidad": None,
                "non_cash_current_assets": None,
                "margen_convergencia": None,
                "margen_ebit_convergencia": 0.07
            },
            "financial_balance_rubio": {
                "first_year": {
                    "date": "2024-12-31",
                    "efectivo_y_equivalentes_al_efectivo": 0.0,
                    "total_activos_corrientes": 0.0,
                    "total_activos_no_corrientes": 0.0,
                    "deuda_financiera": 0,
                    "total_de_pasivos_corrientes": 0.0,
                    "deuda_financiera_pasivo": 0.0,
                    "total_de_pasivos_no_corrientes": 0,
                    "capital_emitido": 0.0,
                    "ganancias_acumuladas": 0.0,
                    "patrimonio_total": 0.0,
                    "activos_circulante": 0.0,
                    "activos_fijos_netos": 0.0,
                    "otros_activos_no_circulantes": 0.0,
                    "total_activos": 0.0,
                    "pasivos_circulantes": 0.0,
                    "otros_pasivos_no_circulantes": 0.0,
                    "total_pasivos": 0.0,
                    "otras_reservas": 0.0,
                    "total_de_patrimonio_y_pasivos": 0.0,
                    "revolving": 0,
                    "patrimonio_atribuible_a_los_propietarios_de_la_controladora": 0
                },
                "second_year": {
                    "date": "2023-12-31",
                    "efectivo_y_equivalentes_al_efectivo": 0.0,
                    "total_activos_corrientes": 0.0,
                    "total_activos_no_corrientes": 0.0,
                    "deuda_financiera": 0,
                    "total_de_pasivos_corrientes": 0.0,
                    "deuda_financiera_pasivo": 0.0,
                    "total_de_pasivos_no_corrientes": 0,
                    "capital_emitido": 0.0,
                    "ganancias_acumuladas": 0.0,
                    "patrimonio_total": 0.0,
                    "activos_circulante": 0.0,
                    "activos_fijos_netos": 0.0,
                    "otros_activos_no_circulantes": 0.0,
                    "total_activos": 0.0,
                    "pasivos_circulantes": 0.0,
                    "otros_pasivos_no_circulantes": 0.0,
                    "total_pasivos": 0.0,
                    "otras_reservas": 0.0,
                    "total_de_patrimonio_y_pasivos": 0.0,
                    "revolving": 0,
                    "patrimonio_atribuible_a_los_propietarios_de_la_controladora": 0
                },
                "third_year": {
                    "date": "2022-12-31",
                    "efectivo_y_equivalentes_al_efectivo": 0.0,
                    "total_activos_corrientes": 0.0,
                    "total_activos_no_corrientes": 0.0,
                    "deuda_financiera": 0,
                    "total_de_pasivos_corrientes": 0.0,
                    "deuda_financiera_pasivo": 0.0,
                    "total_de_pasivos_no_corrientes": 0,
                    "capital_emitido": 0.0,
                    "ganancias_acumuladas": 0.0,
                    "patrimonio_total": 0.0,
                    "activos_circulante": 0.0,
                    "activos_fijos_netos": 0.0,
                    "otros_activos_no_circulantes": 0.0,
                    "total_activos": 0.0,
                    "pasivos_circulantes": 0.0,
                    "otros_pasivos_no_circulantes": 0.0,
                    "total_pasivos": 0.0,
                    "otras_reservas": 0.0,
                    "total_de_patrimonio_y_pasivos": 0.0,
                    "revolving": 0,
                    "patrimonio_atribuible_a_los_propietarios_de_la_controladora": 0
                }
            }
        },
        "check_inputs": True,
        "missing_key": None,
        "f29_results_input": None
    }

    # -----------------------------------------------------------------------
    # EJEMPLO DE CÓMO ASIGNAR LO QUE LLEGA:
    # -----------------------------------------------------------------------
    for partial_data in partial_results_list:
        # Si GPT devolvió algo como: {"f29_partial": {"BaseImponible": 405000}}
        if "f29_partial" in partial_data:
            f29 = partial_data["f29_partial"]
            base_imp = f29.get("BaseImponible", 0)
            # Ejemplo: sumamos la base imponible en net_revenue del first_year
            final_data["quantitative_input"]["statement_income"]["first_year"]["net_revenue"] += base_imp

            # Si hay otros campos en f29, vas sumando o asignando según tu criterio
            # retImp = f29.get("RetImpUnicoTrab", 0)
            # ...
        
        # Si GPT devolvió algo como: {"f22_partial": {"AñoTributario": 2024, "RentaLiquidaImponible": 5000000}}
        if "f22_partial" in partial_data:
            f22 = partial_data["f22_partial"]
            anio = f22.get("AñoTributario", 0)
            # Decide en cuál "year" del JSON lo guardas
            # Si anio == 2024 => "first_year"
            # if anio == 2023 => "second_year"
            # etc.
            # Ejemplo:
            if anio == 2024:
                final_data["quantitative_input"]["statement_income"]["first_year"]["net_income"] += f22.get("RentaLiquidaImponible", 0)
            elif anio == 2023:
                final_data["quantitative_input"]["statement_income"]["second_year"]["net_income"] += f22.get("RentaLiquidaImponible", 0)
            # etc.

        # Y así con balances u otras secciones que detectes en partial_data.
        # "balance_partial": { ... }
        # final_data["quantitative_input"]["financial_balance_rubio"]["first_year"]["activos_fijos_netos"] = ...
        # ...
    
    return final_data

# ============================================================================
# 4) FLUJO PRINCIPAL: OCR -> PROCESAMIENTO EN PARALELO -> COMBINACIÓN
# ============================================================================
def process_pdf(ruta_pdf):
    """
    1) Extrae el PDF con Document Intelligence
    2) Procesa cada página en paralelo con GPT
    3) Combina resultados
    4) Devuelve JSON final y lo guarda
    """
    pdf_data = extract_text_from_pdf(ruta_pdf, AZURE_CONFIG["api_key"], AZURE_CONFIG["endpoint"])
    if not pdf_data:
        logger.error("Error extrayendo PDF.")
        return None

    text_by_page = pdf_data["text_by_page"]
    logger.info(f"Hay {len(text_by_page)} páginas detectadas.")

    # Crear cliente OpenAI
    openai_client = AzureOpenAI(
        api_version=AZURE_OPENAI_CONFIG["api_version"],
        azure_endpoint=AZURE_OPENAI_CONFIG["endpoint"],
        api_key=AZURE_OPENAI_CONFIG["api_key"]
    )

    # -----------------------------------------------------------------------
    # PROCESAR PÁGINAS EN PARALELO
    # -----------------------------------------------------------------------
    partial_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        futures = {
            executor.submit(extract_data_from_page, i+1, pg_text, openai_client): i+1
            for i, pg_text in enumerate(text_by_page)
        }

        for future in as_completed(futures):
            page_index = futures[future]
            try:
                result = future.result()
                partial_results.append(result)
                logger.info(f"Página {page_index} procesada OK.")
            except Exception as e:
                logger.error(f"Error en página {page_index}: {e}")

    # -----------------------------------------------------------------------
    # COMBINAR RESULTADOS
    # -----------------------------------------------------------------------
    final_data = combine_partial_results(partial_results)

    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/final_output_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)

    logger.info(f"JSON final guardado en: {output_file}")
    return final_data

# ============================================================================
# 5) EJECUCIÓN POR LÍNEA DE COMANDOS
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Uso: python pdf_ocr_parallel.py /ruta/al/archivo.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = process_pdf(pdf_path)
    if result:
        logger.info("¡Proceso completado con éxito!")
    else:
        logger.error("Hubo un error en el procesamiento.")
        sys.exit(1)
