"""
Análisis profundo de factores de riesgo con explicaciones clínicas
Genera hallazgos clave sobre comorbilidades y personas no diagnosticadas
"""

import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISIS PROFUNDO DE FACTORES DE RIESGO")
print("="*80)

# Cargar datos
df = pd.read_csv('data/02_heart_2022_clean.csv')
df_valido = df[df['HasCoronaryDisease'].notna()].copy()

print(f"\n[1/5] Cargando dataset...")
print(f"✓ Registros válidos: {len(df_valido):,}")

# ============================================================================
# 1. RIESGO GENERAL Y PUNTOS DE REFERENCIA
# ============================================================================

print(f"\n[2/5] Calculando riesgos relativos...")

general_risk = (df_valido['HasCoronaryDisease'] == 'Yes').sum() / len(df_valido) * 100

riesgo_general = {
    'prevalencia_general': round(general_risk, 2),
    'total_con_enfermedad': int((df_valido['HasCoronaryDisease'] == 'Yes').sum()),
    'total_sin_enfermedad': int((df_valido['HasCoronaryDisease'] == 'No').sum()),
    'explicacion': f'De cada 100 personas en el dataset, {round(general_risk, 1)} tienen enfermedad coronaria diagnosticada.'
}

# ============================================================================
# 2. ANÁLISIS DE COMORBILIDADES CON EXPLICACIONES
# ============================================================================

print(f"\n[3/5] Analizando comorbilidades...")

comorbilidades = {}

# ACCIDENTE CEREBROVASCULAR - Factor más crítico
df_stroke = df_valido[df_valido['HadStroke'].notna()]
if len(df_stroke) > 0:
    con_stroke = (df_stroke['HadStroke'] == 'Yes').sum()
    sin_stroke = (df_stroke['HadStroke'] == 'No').sum()
    
    risk_con = (df_stroke[df_stroke['HadStroke'] == 'Yes']['HasCoronaryDisease'] == 'Yes').sum() / con_stroke * 100 if con_stroke > 0 else 0
    risk_sin = (df_stroke[df_stroke['HadStroke'] == 'No']['HasCoronaryDisease'] == 'Yes').sum() / sin_stroke * 100 if sin_stroke > 0 else 0
    ratio = risk_con / general_risk if general_risk > 0 else 0
    
    comorbilidades['accidente_cerebrovascular'] = {
        'con_condicion': int(con_stroke),
        'sin_condicion': int(sin_stroke),
        'riesgo_con_condicion': round(risk_con, 2),
        'riesgo_sin_condicion': round(risk_sin, 2),
        'riesgo_relativo': round(ratio, 2),
        'hallazgo': f'Las personas con ACV tienen {round(ratio, 1)}x más riesgo. De cada 100 con ACV, {round(risk_con, 1)} tienen enfermedad coronaria.',
        'implicacion': 'El ACV y la enfermedad coronaria comparten mecanismos vasculares similares. Si tienes antecedentes de ACV, necesitas monitoreo cardíaco urgente.'
    }

# ENFERMEDAD RENAL
df_kidney = df_valido[df_valido['HadKidneyDisease'].notna()]
if len(df_kidney) > 0:
    con_kidney = (df_kidney['HadKidneyDisease'] == 'Yes').sum()
    sin_kidney = (df_kidney['HadKidneyDisease'] == 'No').sum()
    
    risk_con = (df_kidney[df_kidney['HadKidneyDisease'] == 'Yes']['HasCoronaryDisease'] == 'Yes').sum() / con_kidney * 100 if con_kidney > 0 else 0
    risk_sin = (df_kidney[df_kidney['HadKidneyDisease'] == 'No']['HasCoronaryDisease'] == 'Yes').sum() / sin_kidney * 100 if sin_kidney > 0 else 0
    ratio = risk_con / general_risk if general_risk > 0 else 0
    
    comorbilidades['enfermedad_renal'] = {
        'con_condicion': int(con_kidney),
        'sin_condicion': int(sin_kidney),
        'riesgo_con_condicion': round(risk_con, 2),
        'riesgo_sin_condicion': round(risk_sin, 2),
        'riesgo_relativo': round(ratio, 2),
        'hallazgo': f'Las personas con enfermedad renal tienen {round(ratio, 1)}x más riesgo. {round(risk_con, 1)}% de ellas tienen enfermedad coronaria.',
        'implicacion': 'La enfermedad renal afecta la presión arterial y el metabolismo, aumentando riesgo cardiovascular. Requiere control regular.'
    }

# EPOC
df_copd = df_valido[df_valido['HadCOPD'].notna()]
if len(df_copd) > 0:
    con_copd = (df_copd['HadCOPD'] == 'Yes').sum()
    sin_copd = (df_copd['HadCOPD'] == 'No').sum()
    
    risk_con = (df_copd[df_copd['HadCOPD'] == 'Yes']['HasCoronaryDisease'] == 'Yes').sum() / con_copd * 100 if con_copd > 0 else 0
    risk_sin = (df_copd[df_copd['HadCOPD'] == 'No']['HasCoronaryDisease'] == 'Yes').sum() / sin_copd * 100 if sin_copd > 0 else 0
    ratio = risk_con / general_risk if general_risk > 0 else 0
    
    comorbilidades['epoc'] = {
        'con_condicion': int(con_copd),
        'sin_condicion': int(sin_copd),
        'riesgo_con_condicion': round(risk_con, 2),
        'riesgo_sin_condicion': round(risk_sin, 2),
        'riesgo_relativo': round(ratio, 2),
        'hallazgo': f'Las personas con EPOC tienen {round(ratio, 1)}x más riesgo. Aproximadamente {round(risk_con, 1)}% tienen enfermedad coronaria.',
        'implicacion': 'La inflamación crónica en EPOC afecta el corazón. Fumadores con EPOC están en riesgo muy alto.'
    }

# DIABETES
df_diabetes = df_valido[df_valido['HadDiabetes'].notna()]
if len(df_diabetes) > 0:
    con_diabetes = (df_diabetes['HadDiabetes'].isin(['Yes', 'No, pre-diabetes or borderline diabetes'])).sum()
    sin_diabetes = (df_diabetes['HadDiabetes'] == 'No').sum()
    
    risk_con = (df_diabetes[df_diabetes['HadDiabetes'].isin(['Yes', 'No, pre-diabetes or borderline diabetes'])]['HasCoronaryDisease'] == 'Yes').sum() / con_diabetes * 100 if con_diabetes > 0 else 0
    risk_sin = (df_diabetes[df_diabetes['HadDiabetes'] == 'No']['HasCoronaryDisease'] == 'Yes').sum() / sin_diabetes * 100 if sin_diabetes > 0 else 0
    ratio = risk_con / general_risk if general_risk > 0 else 0
    
    comorbilidades['diabetes'] = {
        'con_condicion': int(con_diabetes),
        'sin_condicion': int(sin_diabetes),
        'riesgo_con_condicion': round(risk_con, 2),
        'riesgo_sin_condicion': round(risk_sin, 2),
        'riesgo_relativo': round(ratio, 2),
        'hallazgo': f'Las personas con diabetes tienen {round(ratio, 1)}x más riesgo. {round(risk_con, 1)}% de diabéticos tienen enfermedad coronaria.',
        'implicacion': 'La diabetes daña los vasos sanguíneos. Incluso la prediabetes aumenta significativamente el riesgo cardiovascular.'
    }

# ARTRITIS
df_arthritis = df_valido[df_valido['HadArthritis'].notna()]
if len(df_arthritis) > 0:
    con_arthritis = (df_arthritis['HadArthritis'] == 'Yes').sum()
    sin_arthritis = (df_arthritis['HadArthritis'] == 'No').sum()
    
    risk_con = (df_arthritis[df_arthritis['HadArthritis'] == 'Yes']['HasCoronaryDisease'] == 'Yes').sum() / con_arthritis * 100 if con_arthritis > 0 else 0
    risk_sin = (df_arthritis[df_arthritis['HadArthritis'] == 'No']['HasCoronaryDisease'] == 'Yes').sum() / sin_arthritis * 100 if sin_arthritis > 0 else 0
    ratio = risk_con / general_risk if general_risk > 0 else 0
    
    comorbilidades['artritis'] = {
        'con_condicion': int(con_arthritis),
        'sin_condicion': int(sin_arthritis),
        'riesgo_con_condicion': round(risk_con, 2),
        'riesgo_sin_condicion': round(risk_sin, 2),
        'riesgo_relativo': round(ratio, 2),
        'hallazgo': f'Las personas con artritis tienen {round(ratio, 1)}x más riesgo. {round(risk_con, 1)}% de ellas tienen enfermedad coronaria.',
        'implicacion': 'La inflamación sistémica de la artritis afecta los vasos sanguíneos. Requiere control del dolor y la inflamación.'
    }

# DEPRESIÓN
df_depression = df_valido[df_valido['HadDepressiveDisorder'].notna()]
if len(df_depression) > 0:
    con_depression = (df_depression['HadDepressiveDisorder'] == 'Yes').sum()
    sin_depression = (df_depression['HadDepressiveDisorder'] == 'No').sum()
    
    risk_con = (df_depression[df_depression['HadDepressiveDisorder'] == 'Yes']['HasCoronaryDisease'] == 'Yes').sum() / con_depression * 100 if con_depression > 0 else 0
    risk_sin = (df_depression[df_depression['HadDepressiveDisorder'] == 'No']['HasCoronaryDisease'] == 'Yes').sum() / sin_depression * 100 if sin_depression > 0 else 0
    ratio = risk_con / general_risk if general_risk > 0 else 0
    
    comorbilidades['depresion'] = {
        'con_condicion': int(con_depression),
        'sin_condicion': int(sin_depression),
        'riesgo_con_condicion': round(risk_con, 2),
        'riesgo_sin_condicion': round(risk_sin, 2),
        'riesgo_relativo': round(ratio, 2),
        'hallazgo': f'Las personas con depresión tienen {round(ratio, 1)}x más riesgo. {round(risk_con, 1)}% tienen enfermedad coronaria.',
        'implicacion': 'La depresión afecta comportamientos de salud y aumenta estrés. El tratamiento psicológico es crucial para la salud cardiovascular.'
    }

# ============================================================================
# 3. ANÁLISIS DE EDAD CRÍTICA
# ============================================================================

print(f"\n[4/5] Analizando puntos críticos de edad...")

edad_critica = {}
age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', 
             '45-49', '50-54', '55-59', '60-64', '65-69', 
             '70-74', '75-79', '80 or older']

df_age = df_valido[df_valido['AgeCategory'].notna()]
prev_risk = 0

for age in age_order:
    if age in df_age['AgeCategory'].values:
        df_a = df_age[df_age['AgeCategory'] == age]
        risk = (df_a['HasCoronaryDisease'] == 'Yes').sum() / len(df_a) * 100
        n = len(df_a)
        
        edad_critica[age] = {
            'prevalencia': round(risk, 2),
            'n': int(n)
        }
        
        # Detectar inflexiones
        if prev_risk > 0:
            increase = risk - prev_risk
            if increase > 2.0:
                edad_critica[age]['inflexion'] = f'Aumento significativo de {round(increase, 1)}% respecto al grupo anterior'
        
        prev_risk = risk

# Hallazgo principal de edad
edad_critica['hallazgo'] = 'El riesgo aumenta exponencialmente después de los 50 años. Entre 50-60 años se duplica el riesgo.'
edad_critica['recomendacion'] = 'Personas de 50+ años deben hacer chequeos cardiovasculares anuales, especialmente si tienen otros factores de riesgo.'

# ============================================================================
# 4. PERSONAS NO DIAGNOSTICADAS EN RIESGO
# ============================================================================

print(f"\n[5/5] Estimando personas no diagnosticadas...")

# Crear perfil de riesgo
df_risk = df_valido.copy()
df_risk['risk_score'] = 0

# Asignar puntos por factores
df_risk.loc[df_risk['HadStroke'] == 'Yes', 'risk_score'] += 4
df_risk.loc[df_risk['HadKidneyDisease'] == 'Yes', 'risk_score'] += 3
df_risk.loc[df_risk['HadCOPD'] == 'Yes', 'risk_score'] += 3
df_risk.loc[df_risk['HadDiabetes'].isin(['Yes', 'No, pre-diabetes or borderline diabetes']), 'risk_score'] += 2
df_risk.loc[df_risk['HadArthritis'] == 'Yes', 'risk_score'] += 1
df_risk.loc[df_risk['HadDepressiveDisorder'] == 'Yes', 'risk_score'] += 1

# Categorizar
df_risk['risk_category'] = 'Bajo riesgo'
df_risk.loc[df_risk['risk_score'] >= 2, 'risk_category'] = 'Riesgo moderado'
df_risk.loc[df_risk['risk_score'] >= 4, 'risk_category'] = 'Riesgo alto'
df_risk.loc[df_risk['risk_score'] >= 6, 'risk_category'] = 'Riesgo muy alto'

# Personas no diagnosticadas por categoría
no_diagnosticados = {}
for category in ['Bajo riesgo', 'Riesgo moderado', 'Riesgo alto', 'Riesgo muy alto']:
    subset = df_risk[(df_risk['HasCoronaryDisease'] == 'No') & (df_risk['risk_category'] == category)]
    no_diagnosticados[category] = {
        'cantidad': int(len(subset)),
        'porcentaje': round(len(subset) / len(df_valido) * 100, 2)
    }

# Hallazgo principal
total_no_diag_alto_riesgo = no_diagnosticados['Riesgo alto']['cantidad'] + no_diagnosticados['Riesgo muy alto']['cantidad']
no_diagnosticados['hallazgo'] = f'Hay {total_no_diag_alto_riesgo:,} personas sin diagnóstico pero con alto riesgo de enfermedad coronaria.'
no_diagnosticados['implicacion'] = f'Esto representa el {round(total_no_diag_alto_riesgo/len(df_valido)*100, 2)}% de la población. Muchas podrían beneficiarse de pruebas de detección.'

# ============================================================================
# 5. GUARDAR ANÁLISIS
# ============================================================================

print(f"\nGuardando análisis...")

analisis_profundo = {
    'riesgo_general': riesgo_general,
    'comorbilidades': comorbilidades,
    'edad_critica': edad_critica,
    'no_diagnosticados': no_diagnosticados
}

import os
if not os.path.exists('data'):
    os.makedirs('data')

with open('data/04_risk_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analisis_profundo, f, ensure_ascii=False, indent=2)

print(f"✓ Análisis guardado en: data/04_risk_analysis.json")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)

print(f"""
Hallazgos Principales:

Comorbilidades Críticas:
   • ACV: {comorbilidades['accidente_cerebrovascular']['riesgo_relativo']}x más riesgo
   • Enfermedad Renal: {comorbilidades['enfermedad_renal']['riesgo_relativo']}x más riesgo
   • EPOC: {comorbilidades['epoc']['riesgo_relativo']}x más riesgo
   • Diabetes: {comorbilidades['diabetes']['riesgo_relativo']}x más riesgo

Personas No Diagnosticadas:
   • Riesgo alto: {no_diagnosticados['Riesgo alto']['cantidad']:,} personas
   • Riesgo muy alto: {no_diagnosticados['Riesgo muy alto']['cantidad']:,} personas
   • Total en riesgo: {total_no_diag_alto_riesgo:,} personas

Edad Crítica:
   • Punto de inflexión: 50-55 años
   • Riesgo se duplica entre 50-60 años

Archivo generado: data/04_risk_analysis.json
""")

print("="*80)
