"""
Visualizaciones impresionantes para factores de riesgo
Genera: Heatmap, gráficos de burbujas, timeline de edad
"""

import pandas as pd
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERANDO VISUALIZACIONES  DE FACTORES DE RIESGO")
print("="*80)

# Cargar datos
df = pd.read_csv('data/02_heart_2022_clean.csv')
df_valido = df[df['HasCoronaryDisease'].notna()].copy()

print(f"\n[1/4] Preparando datos...")



# ============================================================================
# 1. HEATMAP: Diferencia de riesgo (factor - referencia)
# ============================================================================

print("\n[2/4] Generando matriz de riesgo (Heatmap) con diferencias...")

age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', 
             '45-49', '50-54', '55-59', '60-64', '65-69', 
             '70-74', '75-79', '80 or older']

# ---- Mapeos correctos ----
mapeo_alcohol = {'Frequent': 'Yes', 'Heavy': 'Yes', 'Occasional': 'Yes', 'No': 'No'}
mapeo_fumar = {'Every day': 'Yes', 'Former': 'Yes', 'Some days': 'Yes', 
               'Smoker (Unspecified)': 'Yes', 'Never': 'No'}

df_valido['Tmp_Alcohol'] = df_valido['AlcoholStatus'].map(mapeo_alcohol).replace({np.nan: None})
df_valido['Tmp_Fumar'] = df_valido['SmokerStatus'].map(mapeo_fumar).replace({np.nan: None})

# Sueño saludable
df_valido['Tmp_SleepHours'] = np.where(
    df_valido['SleepHours'].between(7, 9), 'Yes',
    np.where(df_valido['SleepHours'].isna(), None, 'No')
)

# IMC (creamos columna base con tres categorías para luego calcular referencia)
bins_bmi_completo = [0, 18.5, 27, float('inf')]
labels_bmi_completo = ['Desnutrición', 'Sin Sobrepeso', 'Obesidad']
df_valido['Tmp_BMI_cat'] = pd.cut(df_valido['BMI'], bins=bins_bmi_completo, labels=labels_bmi_completo, right=False)

# ---- Factores de riesgo y sus referencias ----
risk_factors = {
    'Obesidad': {
        'col': 'Tmp_BMI_cat', 'val_riesgo': 'Obesidad',
        'ref_col': 'Tmp_BMI_cat', 'ref_val': 'Sin Sobrepeso'
    },
    'Desnutrición': {
        'col': 'Tmp_BMI_cat', 'val_riesgo': 'Desnutrición',
        'ref_col': 'Tmp_BMI_cat', 'ref_val': 'Sin Sobrepeso'
    },
    'Prediabetes': {
        'col': 'HadDiabetes', 'val_riesgo': 'No, pre-diabetes or borderline diabetes',
        'ref_col': 'HadDiabetes', 'ref_val': 'No'
    },
    'Diabetes': {
        'col': 'HadDiabetes', 'val_riesgo': 'Yes',
        'ref_col': 'HadDiabetes', 'ref_val': 'No'
    },
    'ACV': {
        'col': 'HadStroke', 'val_riesgo': 'Yes',
        'ref_col': 'HadStroke', 'ref_val': 'No'
    },
    'EPOC': {
        'col': 'HadCOPD', 'val_riesgo': 'Yes',
        'ref_col': 'HadCOPD', 'ref_val': 'No'
    },
    'Enfermedad Renal': {
        'col': 'HadKidneyDisease', 'val_riesgo': 'Yes',
        'ref_col': 'HadKidneyDisease', 'ref_val': 'No'
    },
    'Artritis': {
        'col': 'HadArthritis', 'val_riesgo': 'Yes',
        'ref_col': 'HadArthritis', 'ref_val': 'No'
    },
    'Asma': {
        'col': 'HadAsthma', 'val_riesgo': 'Yes',
        'ref_col': 'HadAsthma', 'ref_val': 'No'
    },
    'Cáncer de Piel': {
        'col': 'HadSkinCancer', 'val_riesgo': 'Yes',
        'ref_col': 'HadSkinCancer', 'ref_val': 'No'
    },
    'Sedentario': {
        'col': 'PhysicalActivities', 'val_riesgo': 'No',
        'ref_col': 'PhysicalActivities', 'ref_val': 'Yes'
    },
    'Consume Alcohol': {
        'col': 'Tmp_Alcohol', 'val_riesgo': 'Yes',
        'ref_col': 'Tmp_Alcohol', 'ref_val': 'No'
    },
    'Fuma': {
        'col': 'Tmp_Fumar', 'val_riesgo': 'Yes',
        'ref_col': 'Tmp_Fumar', 'ref_val': 'No'
    },
    'No Duerme Bien': {
        'col': 'Tmp_SleepHours', 'val_riesgo': 'No',
        'ref_col': 'Tmp_SleepHours', 'ref_val': 'Yes'
    },
    'Depresión': {
        'col': 'HadDepressiveDisorder', 'val_riesgo': 'Yes',
        'ref_col': 'HadDepressiveDisorder', 'ref_val': 'No'
    }
    
    
}

def get_risk(df, col, val):
    sub = df[df[col] == val]
    if len(sub) == 0:
        return None
    

    return (sub['HasCoronaryDisease'] == 'Yes').sum() / len(sub) * 100

heatmap_data = []

for age in age_order:
    df_age = df_valido[df_valido['AgeCategory'] == age]
    if len(df_age) == 0:
        continue
    
    for factor_name, specs in risk_factors.items():
        riesgo_con = get_risk(df_age, specs['col'], specs['val_riesgo'])
        riesgo_sin = get_risk(df_age, specs['ref_col'], specs['ref_val'])
        
    
        if riesgo_con is not None and riesgo_sin is not None:
            diff = round(riesgo_con - riesgo_sin, 1)
            heatmap_data.append({
                'age': age,
                'factor': factor_name,
                'risk': diff,
                'count': len(df_age[df_age[specs['col']] == specs['val_riesgo']])
            })


# ============================================================================
# 2. BUBBLE CHART: Tamaño = población, Color = riesgo
# ============================================================================

print(f"\n[3/4] Generando datos para gráfico de burbujas...")

bubble_data = []

# Por comorbilidad
comorbidities = {
    'Sin Diabetes': ('HadDiabetes', 'No'),
    'Prediabetes': ('HadDiabetes', 'No, pre-diabetes or borderline diabetes'),
    'Diabetes': ('HadDiabetes', 'Yes'),
    'Sin ACV': ('HadStroke', 'No'),
    'Con ACV': ('HadStroke', 'Yes'),
    'Sin EPOC': ('HadCOPD', 'No'),
    'Con EPOC': ('HadCOPD', 'Yes'),
    'Sin Enfermedad Renal': ('HadKidneyDisease', 'No'),
    'Con Enfermedad Renal': ('HadKidneyDisease', 'Yes'),
    'Sin Artritis': ('HadArthritis', 'No'),
    'Con Artritis': ('HadArthritis', 'Yes'),
    
}

for cond_name, (col, val) in comorbidities.items():
    df_cond = df_valido[df_valido[col] == val]
    if len(df_cond) > 0:
        risk = (df_cond['HasCoronaryDisease'] == 'Yes').sum() / len(df_cond) * 100
        bubble_data.append({
            'name': cond_name,
            'population': int(len(df_cond)),
            'risk': round(risk, 1),
            'cases': int((df_cond['HasCoronaryDisease'] == 'Yes').sum())
        })

# ============================================================================
# 3. TIMELINE: Riesgo por edad (animado)
# ============================================================================

print(f"\n[4/4] Generando timeline de edad...")

timeline_data = []

for age in age_order:
    df_age = df_valido[df_valido['AgeCategory'] == age]
    if len(df_age) > 0:
        risk = (df_age['HasCoronaryDisease'] == 'Yes').sum() / len(df_age) * 100
        cases = (df_age['HasCoronaryDisease'] == 'Yes').sum()
        
        # Extraer rango numérico para ordenar
        if 'older' in age:
            age_num = 85
            age_label = '80+'
        else:
            age_num = int(age.split('-')[0])
            age_label = f"{age_num}-{age_num+4}"
        
        timeline_data.append({
            'age_num': age_num,
            'age_label': age_label,
            'risk': round(risk, 2),
            'cases': int(cases),
            'population': len(df_age),
            'risk_increase': 0  # Se calcula después
        })

# Calcular aumentos de riesgo
for i in range(1, len(timeline_data)):
    increase = timeline_data[i]['risk'] - timeline_data[i-1]['risk']
    timeline_data[i]['risk_increase'] = round(increase, 2)

# ============================================================================
# 4. MATRIZ DE COMORBILIDADES: Qué factores se presentan juntos
# ============================================================================

print(f"\nGenerando matriz de comorbilidades...")

comorbidity_matrix = {}

conditions = [
    ('HadDiabetes', 'Diabetes'),
    ('HadStroke', 'ACV'),
    ('HadCOPD', 'EPOC'),
    ('HadKidneyDisease', 'Enfermedad Renal'),
    ('HadArthritis', 'Artritis'),
    ('HadDepressiveDisorder', 'Depresión'),
    ('Tmp_Fumar', 'Fumador')
]

for i, (col1, name1) in enumerate(conditions):
    for j, (col2, name2) in enumerate(conditions):
        if i < j:   # Solo combinaciones distintas, sin repetir la misma condición
            df_both = df_valido[
                (df_valido[col1] == 'Yes') & 
                (df_valido[col2] == 'Yes')
            ]
            
            if len(df_both) > 0:
                risk = (df_both['HasCoronaryDisease'] == 'Yes').sum() / len(df_both) * 100
                key = f"{name1} + {name2}"
                comorbidity_matrix[key] = {
                    'population': int(len(df_both)),
                    'risk': round(risk, 1),
                    'cases': int((df_both['HasCoronaryDisease'] == 'Yes').sum())
                }



# ============================================================================
# 5. GUARDAR DATOS
# ============================================================================

print(f"\nGuardando datos...")

visualizaciones = {
    'heatmap': heatmap_data,
    'bubbles': sorted(bubble_data, key=lambda x: x['risk'], reverse=True),
    'timeline': timeline_data,
    'comorbidity_matrix': comorbidity_matrix,
    'resumen': {
        'total_registros': len(df_valido),
        'total_con_enfermedad': int((df_valido['HasCoronaryDisease'] == 'Yes').sum()),
        'prevalencia_general': round((df_valido['HasCoronaryDisease'] == 'Yes').sum() / len(df_valido) * 100, 2),
        'factor_mas_riesgoso': max(bubble_data, key=lambda x: x['risk'])['name'],
        'riesgo_maximo': max(bubble_data, key=lambda x: x['risk'])['risk'],
        'edad_critica': '50-54 años (punto de inflexión)',
        'comorbilidad_mas_peligrosa': max(comorbidity_matrix.items(), key=lambda x: x[1]['risk'])[0]
    }
}

import os
if not os.path.exists('data'):
    os.makedirs('data')

with open('data/05_advanced_risks.json', 'w', encoding='utf-8') as f:
    json.dump(visualizaciones, f, ensure_ascii=False, indent=2)

print(f"✓ Datos guardados en: data/05_advanced_risks.json")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*80)
print("VISUALIZACIONES GENERADAS")
print("="*80)

print(f"""
Datos listos para visualizar:

1. HEATMAP (Matriz de riesgo):
   - {len(heatmap_data)} combinaciones de edad + factor
   - Muestra cómo cada factor aumenta riesgo por edad

2. BUBBLE CHART (Gráfico de burbujas):
   - {len(bubble_data)} condiciones analizadas
   - Tamaño = población, Color = riesgo
   - Factor más riesgoso: {visualizaciones['resumen']['factor_mas_riesgoso']} ({visualizaciones['resumen']['riesgo_maximo']}%)

3. TIMELINE (Línea de tiempo de edad):
   - {len(timeline_data)} grupos de edad
   - Muestra crecimiento exponencial del riesgo
   - Punto de inflexión: {visualizaciones['resumen']['edad_critica']}

4. MATRIZ DE COMORBILIDADES:
   - {len(comorbidity_matrix)} combinaciones de factores
   - Comorbilidad más peligrosa: {visualizaciones['resumen']['comorbilidad_mas_peligrosa']}

Archivo: data/09_impressive_risk_viz.json
""")

print("="*80)
