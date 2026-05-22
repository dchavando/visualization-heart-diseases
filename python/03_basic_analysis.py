"""
Análisis básico del dataset
Genera gráficos y estadísticas para conocer la estructura de los datos
"""

import os
import pandas as pd
import json
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISIS BÁSICO DEL DATASET")
print("="*80)

# Cargar dataset limpio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'data', '02_heart_2022_clean.csv')
df = pd.read_csv(INPUT_PATH)

print(f"\n[1/5] Cargando dataset...")
print(f"✓ Registros: {len(df):,}")
print(f"✓ Columnas: {df.shape[1]}")

# ============================================================================
# 1. ESTADÍSTICAS GENERALES
# ============================================================================

print(f"\n[2/5] Calculando estadísticas generales...")

# Variable objetivo
df_valido = df[df['HasCoronaryDisease'].notna()]
total_con_enfermedad = (df_valido['HasCoronaryDisease'] == 'Yes').sum()
total_sin_enfermedad = (df_valido['HasCoronaryDisease'] == 'No').sum()
prevalencia = (total_con_enfermedad / len(df_valido)) * 100

estadisticas_generales = {
    'total_registros': int(len(df)),
    'registros_validos': int(len(df_valido)),
    'con_enfermedad_coronaria': int(total_con_enfermedad),
    'sin_enfermedad_coronaria': int(total_sin_enfermedad),
    'prevalencia_porcentaje': round(prevalencia, 2),
    'total_columnas': df.shape[1],
    'valores_nulos_totales': int(df.isnull().sum().sum()),
    'porcentaje_completitud': round((1 - df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100, 2)
}

# ============================================================================
# 2. ANÁLISIS DEMOGRÁFICO EXPANDIDO
# ============================================================================

print(f"\n[3/5] Analizando demografía...")

demografica = {}

# Por sexo
df_sexo = df[df['Sex'].notna()]
if len(df_sexo) > 0:
    por_sexo = {}
    for sexo in df_sexo['Sex'].unique():
        df_s = df_sexo[df_sexo['Sex'] == sexo]
        por_sexo[str(sexo)] = {
            'total': int(len(df_s)),
            'porcentaje': round((len(df_s) / len(df_sexo)) * 100, 2)
        }
    demografica['por_sexo'] = por_sexo





# Por edad
df_edad = df[df['AgeCategory'].notna()]
if len(df_edad) > 0:
    por_edad = {}
    edad_orden = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
    for edad in edad_orden:
        if edad in df_edad['AgeCategory'].values:
            df_e = df_edad[df_edad['AgeCategory'] == edad]
            por_edad[str(edad)] = {
                'total': int(len(df_e)),
                'porcentaje': round((len(df_e) / len(df_edad)) * 100, 2)
            }
    demografica['por_edad'] = por_edad


# Por salud general
df_salud = df[df['GeneralHealth'].notna()]
if len(df_salud) > 0:
    por_salud = {}
    salud_orden = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']
    for salud in salud_orden:
        if salud in df_salud['GeneralHealth'].values:
            df_h = df_salud[df_salud['GeneralHealth'] == salud]
            por_salud[str(salud)] = {
                'total': int(len(df_h)),
                'porcentaje': round((len(df_h) / len(df_salud)) * 100, 2)
            }
    demografica['por_salud_general'] = por_salud

# Por raza/etnicidad (para la siguiente sección)
df_raza = df[df['RaceEthnicityCategory'].notna()]
if len(df_raza) > 0:
    por_raza = {}
    for raza in sorted(df_raza['RaceEthnicityCategory'].unique()):
        df_r = df_raza[df_raza['RaceEthnicityCategory'] == raza]
        por_raza[str(raza)] = {
            'total': int(len(df_r)),
            'porcentaje': round((len(df_r) / len(df_raza)) * 100, 2)
        }
    demografica['por_raza'] = por_raza


# ============================================================================
# 3. ANÁLISIS DE HÁBITOS Y COMPORTAMIENTOS
# ============================================================================

print(f"\n[4/5] Analizando hábitos...")

habitos = {}

# Actividad física
df_actividad = df[df['PhysicalActivities'].notna()]
if len(df_actividad) > 0:
    actividad_fisica = {}
    for actividad in df_actividad['PhysicalActivities'].unique():
        df_a = df_actividad[df_actividad['PhysicalActivities'] == actividad]
        actividad_fisica[str(actividad)] = {
            'total': int(len(df_a)),
            'porcentaje': round((len(df_a) / len(df_actividad)) * 100, 2)
        }
    habitos['actividad_fisica'] = actividad_fisica

# Tabaquismo
df_tabaco = df[df['SmokerStatus'].notna()]
if len(df_tabaco) > 0:
    tabaquismo = {}
    for tabaco in df_tabaco['SmokerStatus'].unique():
        df_t = df_tabaco[df_tabaco['SmokerStatus'] == tabaco]
        tabaquismo[str(tabaco)] = {
            'total': int(len(df_t)),
            'porcentaje': round((len(df_t) / len(df_tabaco)) * 100, 2)
        }
    habitos['tabaquismo'] = tabaquismo

# Alcohol
df_alcohol = df[df['AlcoholStatus'].notna()]
if len(df_alcohol) > 0:
    consumo_alcohol = {}
    for alcohol in df_alcohol['AlcoholStatus'].unique():
        df_al = df_alcohol[df_alcohol['AlcoholStatus'] == alcohol]
        consumo_alcohol[str(alcohol)] = {
            'total': int(len(df_al)),
            'porcentaje': round((len(df_al) / len(df_alcohol)) * 100, 2)
        }
    habitos['consumo_alcohol'] = consumo_alcohol

# ============================================================================
# 4. ESTADÍSTICAS NUMÉRICAS EXPANDIDAS
# ============================================================================

print(f"\n[5/5] Calculando estadísticas numéricas...")

estadisticas_numericas = {}

# BMI
df_bmi = df[df['BMI'].notna()]
if len(df_bmi) > 0:
    bmi_values = df_bmi['BMI'].values
    estadisticas_numericas['bmi'] = {
        'promedio': round(df_bmi['BMI'].mean(), 2),
        'mediana': round(df_bmi['BMI'].median(), 2),
        'minimo': round(df_bmi['BMI'].min(), 2),
        'maximo': round(df_bmi['BMI'].max(), 2),
        'desv_estandar': round(df_bmi['BMI'].std(), 2),
        'percentil_25': round(np.percentile(bmi_values, 25), 2),
        'percentil_75': round(np.percentile(bmi_values, 75), 2)
    }

# Sueño
df_sueno = df[df['SleepHours'].notna()]
if len(df_sueno) > 0:
    sleep_values = df_sueno['SleepHours'].values
    estadisticas_numericas['sleep_hours'] = {
        'promedio': round(df_sueno['SleepHours'].mean(), 2),
        'mediana': round(df_sueno['SleepHours'].median(), 2),
        'minimo': round(df_sueno['SleepHours'].min(), 2),
        'maximo': round(df_sueno['SleepHours'].max(), 2),
        'desv_estandar': round(df_sueno['SleepHours'].std(), 2),
        'percentil_25': round(np.percentile(sleep_values, 25), 2),
        'percentil_75': round(np.percentile(sleep_values, 75), 2)
    }

# Días de mala salud
df_unhealthy = df[df['UnhealthyDays'].notna()]
if len(df_unhealthy) > 0:
    unhealthy_values = df_unhealthy['UnhealthyDays'].values
    estadisticas_numericas['unhealthy_days'] = {
        'promedio': round(df_unhealthy['UnhealthyDays'].mean(), 2),
        'mediana': round(df_unhealthy['UnhealthyDays'].median(), 2),
        'minimo': round(df_unhealthy['UnhealthyDays'].min(), 2),
        'maximo': round(df_unhealthy['UnhealthyDays'].max(), 2),
        'desv_estandar': round(df_unhealthy['UnhealthyDays'].std(), 2),
        'percentil_25': round(np.percentile(unhealthy_values, 25), 2),
        'percentil_75': round(np.percentile(unhealthy_values, 75), 2)
    }

# ============================================================================
# 5. ANÁLISIS POR ESTADO
# ============================================================================

print(f"\nAnalizando por estado...")

estados_analisis = {}

for estado in df['State'].unique():
    if pd.isna(estado):
        continue
    
    df_estado = df[df['State'] == estado]
    df_estado_valido = df_estado[df_estado['HasCoronaryDisease'].notna()]
    
    if len(df_estado_valido) == 0:
        continue
    
    con_enfermedad = (df_estado_valido['HasCoronaryDisease'] == 'Yes').sum()
    total_estado = len(df_estado_valido)
    
    estados_analisis[estado] = {
        'total_registros': int(total_estado),
        'con_enfermedad': int(con_enfermedad),
        'sin_enfermedad': int(total_estado - con_enfermedad),
        'prevalencia_porcentaje': round((con_enfermedad / total_estado) * 100, 2)
    }

# ============================================================================
# 6. ANÁLISIS DE CONDICIONES CRÓNICAS
# ============================================================================

print(f"\nAnalizando condiciones crónicas...")

condiciones_cronicas = {}

condiciones = {
    'HadStroke': 'Accidente Cerebrovascular',
    'HadAsthma': 'Asma',
    'HadCOPD': 'EPOC',
    'HadDepressiveDisorder': 'Depresión',
    'HadKidneyDisease': 'Enfermedad Renal',
    'HadArthritis': 'Artritis',
    'HadDiabetes': 'Diabetes'
}

for col, nombre in condiciones.items():
    if col in df.columns:
        df_cond = df[df[col].notna()]
        if len(df_cond) > 0:
            con_condicion = (df_cond[col] == 'Yes').sum()
            condiciones_cronicas[nombre] = {
                'total': int(len(df_cond)),
                'con_condicion': int(con_condicion),
                'sin_condicion': int(len(df_cond) - con_condicion),
                'prevalencia_porcentaje': round((con_condicion / len(df_cond)) * 100, 2)
            }

# ============================================================================
# 7. ANÁLISIS DE RELACIONES CON ENFERMEDAD CORONARIA
# ============================================================================

print(f"\nAnalizando relaciones con enfermedad coronaria...")

relaciones = {}

# Enfermedad coronaria por sexo
df_sexo_enfermedad = df_valido[df_valido['Sex'].notna()]
if len(df_sexo_enfermedad) > 0:
    por_sexo_enfermedad = {}
    for sexo in df_sexo_enfermedad['Sex'].unique():
        df_s = df_sexo_enfermedad[df_sexo_enfermedad['Sex'] == sexo]
        con_enf = (df_s['HasCoronaryDisease'] == 'Yes').sum()
        por_sexo_enfermedad[str(sexo)] = {
            'total': int(len(df_s)),
            'con_enfermedad': int(con_enf),
            'sin_enfermedad': int(len(df_s) - con_enf),
            'prevalencia_porcentaje': round((con_enf / len(df_s)) * 100, 2)
        }
    relaciones['por_sexo'] = por_sexo_enfermedad

# ============================================================================
# 8. GUARDAR ANÁLISIS EN JSON
# ============================================================================

print(f"\nGuardando análisis...")

analisis_basico = {
    'estadisticas_generales': estadisticas_generales,
    'demografica': demografica,
    'habitos': habitos,
    'estadisticas_numericas': estadisticas_numericas,
    'por_estado': estados_analisis,
    'condiciones_cronicas': condiciones_cronicas,
    'relaciones': relaciones
}

import os
if not os.path.exists('data'):
    os.makedirs('data')

with open('data/03_basic_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analisis_basico, f, ensure_ascii=False, indent=2)

print(f"✓ Análisis guardado en: data/03_basic_analysis.json")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)

print(f"""
Estadísticas Generales:
   • Total de registros: {estadisticas_generales['total_registros']:,}
   • Registros válidos: {estadisticas_generales['registros_validos']:,}
   • Con enfermedad coronaria: {estadisticas_generales['con_enfermedad_coronaria']:,}
   • Prevalencia: {estadisticas_generales['prevalencia_porcentaje']}%
   • Completitud de datos: {estadisticas_generales['porcentaje_completitud']}%

Análisis Geográfico:
   • Estados analizados: {len(estados_analisis)}

Condiciones Crónicas Analizadas:
   • Total de condiciones: {len(condiciones_cronicas)}

Archivo generado: data/03_basic_analysis.json
""")

print("="*80)
