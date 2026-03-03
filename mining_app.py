# mining_app.py

import pandas as pd
import plotly.express as px
import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))



def ejecutar_mineria_determinista(df: pd.DataFrame):

    # Columnas EXACTAS
    col_pasarela = 'linea_base_tipo_canal_venta_digital_pasarela_pagos'
    col_mercados = 'cierre_cambios_kinesis_nuevos_mercados'
    col_ventas = 'cierre_cambios_kinesis_venta_productos_servicios'

    # Trabajamos solo con 3 columnas
    df_demo = df[[col_pasarela, col_mercados, col_ventas]].copy()

    df_demo[col_pasarela] = df_demo[col_pasarela].fillna('Sin_Respuesta')
    df_demo[col_mercados] = df_demo[col_mercados].fillna('Sin cambios')
    df_demo[col_ventas] = df_demo[col_ventas].fillna('Sin cambios')

    # Condiciones deterministas
    cond_A = (df_demo[col_pasarela] == 'Sin_Respuesta') & (df_demo[col_mercados] == 'Sin cambios')
    cond_C = (df_demo[col_ventas] == 'Sin cambios')
    cond_A_and_C = cond_A & cond_C

    N_total = len(df_demo)
    n_A = int(cond_A.sum())
    n_C = int(cond_C.sum())
    n_A_and_C = int(cond_A_and_C.sum())

    soporte = (n_A_and_C / N_total) * 100 if N_total > 0 else 0
    confianza = (n_A_and_C / n_A) * 100 if n_A > 0 else 0
    lift = (n_A_and_C / n_A) / (n_C / N_total) if (n_A > 0 and n_C > 0) else 0

    llm_payload = {
        "contexto_analisis": "Minería de reglas de asociación (Efecto de la digitalización en ventas).",
        "metricas_duras": {
            "muestra_total_empresas": N_total,
            "empresas_que_cumplen_condicion_inicial": n_A,
            "empresas_que_cumplen_ambas_condiciones": n_A_and_C,
            "soporte_porcentaje": round(soporte, 1),
            "confianza_porcentaje": round(confianza, 1),
            "lift_fuerza_asociacion": round(lift, 2)
        }
    }

    # =====================
    # GRÁFICO 3D
    # =====================
    df_agrupado = df_demo.groupby(
        [col_pasarela, col_mercados, col_ventas]
    ).size().reset_index(name='Cantidad')

    df_agrupado['Patron'] = df_agrupado.apply(
        lambda row: 'Alerta: Efecto dominó (estancamiento)'
        if row[col_pasarela] == 'Sin_Respuesta'
        and row[col_mercados] == 'Sin cambios'
        and row[col_ventas] == 'Sin cambios'
        else 'Otras trayectorias',
        axis=1
    )

    fig = px.scatter_3d(
        df_agrupado,
        x=col_pasarela,
        y=col_mercados,
        z=col_ventas,
        size='Cantidad',
        color='Patron',
        size_max=45,
        opacity=0.9,
        title='<b>El Efecto Dominó: Impacto de la Digitalización en Ventas</b>'
    )

    fig.update_layout(
    template='plotly_white',
    height=1050,   # 🔥 hace el gráfico más alto
    width=1200,   # 🔥 hace el gráfico más ancho
    margin=dict(l=0, r=0, b=0, t=60),
    scene=dict(
        xaxis=dict(title='Pasarela de pagos'),
        yaxis=dict(title='Nuevos mercados'),
        zaxis=dict(title='Cambio en ventas')
    )
)


    fig.update_layout(template='plotly_white')

    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                    "Eres un analista. Interpreta el JSON de métricas y explica en 1-2 párrafos "
                    "qué significa el patrón encontrado. Sé claro y directo. No uses listas."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(llm_payload, ensure_ascii=False)
                }
            ],
        )
        interpretacion_texto = resp.choices[0].message.content
    except Exception:
        m = llm_payload["metricas_duras"]
        interpretacion_texto = (
            f"Se detecta un patrón donde {m['empresas_que_cumplen_ambas_condiciones']} empresas "
            f"cumplen simultáneamente la condición inicial y el estancamiento en ventas. "
            f"El soporte es {m['soporte_porcentaje']}%, con confianza {m['confianza_porcentaje']}% "
            f"y lift {m['lift_fuerza_asociacion']}."
        )

    return {
        "tipo": "grafico",
        "datos_del_grafico": json.loads(fig.to_json()),
        "interpretacion": interpretacion_texto
    }   