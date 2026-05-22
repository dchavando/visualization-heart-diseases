"""
Script de limpieza de datos
Elimina duplicados exactos y estandariza valores
Preserva todos los valores nulos (no se imputan ni se eliminan filas)
Contempla las nuevas columnas agregadas al dataset
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LIMPIEZA DE DATOS - ENFERMEDADES CARDÍACAS")
print("="*80)


# ============================================================================
# 1. CARGA DATASET
# ============================================================================

print("\n[1/5] Cargando dataset...")
if not os.path.exists('data/01_heart_with_nans.csv'):
    print(f"✗ Error: No se encontró el archivo de entrada en: {'data/01_heart_with_nans.csv'}")
    exit(1)

df = pd.read_csv('data/01_heart_with_nans.csv')

print(f"✓ Dataset cargado: {df.shape[0]:,} registros, {df.shape[1]} columnas")
print(f"✓ Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"✓ Valores nulos totales: {df.isnull().sum().sum():,}")



# ============================================================================
# 2. IDENTIFICAR DUPLICADOS
# ============================================================================

print("\n[2/5] Identificando duplicados...")

duplicados_antes = len(df)
# Forzamos un .copy() para desvincularlo del dataframe original y evitar warnings
df_sin_duplicados = df.drop_duplicates().copy()
duplicados_encontrados = duplicados_antes - len(df_sin_duplicados)

if duplicados_encontrados > 0:
    print(f"✓ Duplicados encontrados: {duplicados_encontrados:,}")
    df = df_sin_duplicados
else:
    print(f"✓ No hay duplicados exactos")



# ============================================================================
# 3. LIMPIA Y ESTANDARIZA VALORES
# ============================================================================

print("\n[3/5] Estandarizando valores...")

# Seleccionar solo columnas de tipo objeto (strings)
columnas_categoricas = df.select_dtypes(include=['object']).columns

# Diccionario de estandarización limpia
mapeo_estandarizacion = {
    'no': 'No',
    'yes': 'Yes',
    'don\'t know': "Don't know",
    'dont know': "Don't know",
    'refused': 'Refused'
}

for col in columnas_categoricas:
    # Eliminar espacios en blanco al inicio y al final
    df[col] = df[col].astype(str).str.strip()
    
    # Aplicar mapeo ignorando mayúsculas/minúsculas mediante una transformación temporal a minúsculas
    # Se preservan los NaNs originales mapeados por la lectura de pandas
    df[col] = df[col].str.lower().map(mapeo_estandarizacion).fillna(df[col])
    
    # Restaurar los valores 'nan' de string de vuelta a un objeto NaN real si surgieron en el astype
    df[col] = df[col].replace({'nan': np.nan, 'None': np.nan})


df["BMI"] = np.where((df["BMI"] < 10) | (df["BMI"] > 60), np.nan, df["BMI"])


print(f"✓ Valores estandarizados en {len(columnas_categoricas)} columnas categóricas")





# ============================================================================
# 4. UNIFICA Y CLASIFICA COLUMNAS
# ============================================================================

print("\n[4/5] Analizando y unificando nuevas columnas...")


# --- 1. Dimensiones de Registro, Identificación y Etnia ---
df['SurveyDate'] = pd.to_datetime(df['SurveyDate'].astype(str).str.zfill(8), format='%m%d%Y', errors='coerce')


df['DispositionCode'] = df['DispositionCode'].replace({
    'Completed interview': 'Completed', 
    'Partially completed interview': 'Partially'
})

df["RaceEthnicityCategory"] = df["RaceEthnicityCategory"].replace({
    "White only, Non-Hispanic": "White",
    "Black only, Non-Hispanic": "Black",
    "Multiracial, Non-Hispanic": "Multiracial",
    "Other race only, Non-Hispanic": "Other"
})


# RANGO DE EDADES
mapeo_edades = {
    'Age 18 to 24': '18-24',
    'Age 25 to 29': '25-29',
    'Age 30 to 34': '30-34',
    'Age 35 to 39': '35-39',
    'Age 40 to 44': '40-44',
    'Age 45 to 49': '45-49',
    'Age 50 to 54': '50-54',
    'Age 55 to 59': '55-59',
    'Age 60 to 64': '60-64',
    'Age 65 to 69': '65-69',
    'Age 70 to 74': '70-74',
    'Age 75 to 79': '75-79',
    'Age 80 or older': '80 or older'
}
df['AgeCategory'] = df['AgeCategory'].replace(mapeo_edades)


# --- 2. Perfil de Salud General y Días Activos ---
df['UnhealthyDays'] = np.where((df['PhysicalHealthDays'] == 0) & (df['MentalHealthDays'] == 0), 0, df['PoorHealthDays'])

# --- 3. Control de Chequeos Médicos y Dentales ---
df['LastCheckupTime'] = df['LastCheckupTime'].replace({
    'Within past year (anytime less than 12 months ago)': '< 1 year',
    'Within past 2 years (1 year but less than 2 years ago)': '1 - 2 years',
    'Within past 5 years (2 years but less than 5 years ago)': '2 - 5 years',
    '5 or more years ago': '5+ years'
})

df['LastDentistVisit'] = df['LastDentistVisit'].replace({
    'Less than 1 year ago': '< 1 year',
    '1 year to less than 2 years ago': '1 - 2 years',
    '2 years to less than 5 years ago': '2 - 5 years',
    '5 years or more': '5+ years'
})


# --- 4. Variable Objetivo: HasCoronaryDisease (Directo y Seguro) ---
df["HasCoronaryDisease"] = np.select(
    [
        (df["HadHeartAttack"] == "Yes") | (df["HadAngina"] == "Yes"),
        (df["HadHeartAttack"] == "No") & (df["HadAngina"] == "No")
    ], 
    ["Yes", "No"], 
    default=None
)

# --- 5. Bloque de Discapacidades (Directo) ---
df["DisabilityPhysical"] = np.where(
    df[["DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands"]].isin(["Yes"]).any(axis=1), 1, 
    np.where(df[["DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands"]].isnull().all(axis=1), np.nan, 0)
)

df["DisabilitySensoryCognitive"] = np.where(
    df[["DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating"]].isin(["Yes"]).any(axis=1), 1, 
    np.where(df[["DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating"]].isnull().all(axis=1), np.nan, 0)
)

# --- 6. Perfil de Tabaquismo y Vapeo ---
df["SmokerStatus"] = df["SmokerStatus"].replace({
    "Never smoked": "Never",
    "Former smoker": "Former",
    "Current smoker - now smokes every day": "Every day",
    "Current smoker - now smokes some days": "Some days"
})
df["SmokerStatus"] = np.where(df["SmokerStatus"].isnull() & (df["Ever100Cigarettes"] == "Yes"), "Smoker (Unspecified)", df["SmokerStatus"])

df["ECigaretteStatus"] = df["ECigaretteUsage"].replace({
    "Never used e-cigarettes in my entire life": "Never",
    "Not at all (right now)": "Former",
    "Use them every day": "Every day",
    "Use them some days": "Some days"
})

# --- 7. Bloque de Alcohol ---
df["DrinksPerWeek"] = df["DrinksPerWeek"].replace(99900, np.nan)
df["DrinksPerWeek"] = np.where(df["DrinksPerWeek"] < 100, df["DrinksPerWeek"], df["DrinksPerWeek"] / 100)
df["DrinksPerWeek"] = np.where(df["DrinksPerWeek"] < 0.001, 0, df["DrinksPerWeek"])



# IsDrinker 
df["IsDrinker"] = np.select(
    [
        (df["AlcoholDrinkers"] == "Yes") ,
        (df["AlcoholDrinkers"] == "No")
    ], 
    [1, 0], 
    default=None
)

# AlcoholStatus (Corregido con default=None)
df["AlcoholStatus"] = np.select(
    [
        (df["IsDrinker"] == 0),
        (df["BingeIndicator"] == 2) | (df["DrinksPerWeek"] > 14) | (df["AlcoholRiskIndicator"] == 2),
        (df["DrinksPerWeek"] >= 3) & (df["DrinksPerWeek"] <= 14),
        (df["IsDrinker"] == 1),
    ], 
    ["No", "Heavy", "Frequent", "Occasional"],
    default=None
)

df["AlcoholStatus"] = pd.Categorical(df["AlcoholStatus"], categories=["No", "Occasional", "Frequent", "Heavy"], ordered=True)


# ============================================================================
# ELIMINACIÓN DE COLUMNAS EXCEDENTES (Excluyendo FluVaxLast12)
# ============================================================================
df.drop(
    columns=[
        "HadHeartAttack", "HadAngina",
        "PhysicalHealthDays", "MentalHealthDays", "PoorHealthDays",
        "DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands",
        "DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating",
        "Ever100Cigarettes", "SmokeFrequency", "AgeFirstCigarette", "CigarettesPerDay", "ECigaretteUsage",
        "AlcoholConsumptionDays", "AverageDrinksPerDay", "DaysWithFivePlusDrinks", "MaxDrinksPerOccasion", 
        "BingeIndicator", "AlcoholRiskIndicator", "AlcoholAdvice", "AlcoholDrinkers", "BingeConsumption", 
        "BingeAdvice", "ReduceConsumption",
        "PneumoVaxEver", "TetanusLast10Tdap", "HIVTesting", "HighRiskLastYear"
    ], 
    errors='ignore', 
    inplace=True
)

# ============================================================================
# ACOMODO DE COLUMNAS 
# ============================================================================
columnas_ordenadas = [
    "SurveyDate", "DispositionCode", 
    "State", "Sex", "RaceEthnicityCategory", "AgeCategory", 
    "GeneralHealth", "UnhealthyDays", "SleepHours", 
    "LastCheckupTime", "LastDentistVisit",  "HasCoronaryDisease",  "HadStroke", "HadAsthma", "HadSkinCancer", 
    "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabetes", 
    "LungCancerConcern", "ChestScan", "RemovedTeeth", "PhysicalActivities", "SmokerStatus", "ECigaretteStatus", 
    "AlcoholStatus", "DisabilityPhysical", "DisabilitySensoryCognitive", 
    "HeightInMeters", "WeightInKilograms", "BMI", 
    "CovidPos", 
]

# Filtrar solo las columnas que realmente existan en el DataFrame para evitar KeyErrors accidentales
columnas_finales = [col for col in columnas_ordenadas if col in df.columns]
df = df[columnas_finales]
print("✓ Columnas unificadas, dataset limpio y reordenado correctamente.")





# ============================================================================
# 5. GUARDA DATASET LIMPIO
# ============================================================================

print("\n[5/5] Guardando dataset limpio...")

# Guardar dataset limpio
df.to_csv("data/02_heart_2022_clean.csv", index=False)

print(f"✓ Dataset limpio guardado en: {"data/02_heart_2022_clean.csv"}")
print(f"✓ Registros finales: {len(df):,}")
print(f"✓ Columnas: {df.shape[1]}")
print(f"✓ Valores nulos: {df.isnull().sum().sum():,} (preservados)")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("LIMPIEZA COMPLETADA")
print("="*80)

print(f"""
ESTADÍSTICAS DE LIMPIEZA:
   • Registros iniciales: {duplicados_antes:,}
   • Duplicados eliminados: {duplicados_encontrados:,}
   • Registros finales: {len(df):,}
   • Columnas totales: {df.shape[1]}


Archivo generado exitosamente.

""")
