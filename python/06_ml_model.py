"""
Modelo de Machine Learning para predecir HasCoronaryDisease
Identifica variables más influyentes y realiza predicciones con métricas detalladas
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("MODELO DE MACHINE LEARNING - PREDICCIÓN DE ENFERMEDAD CORONARIA")
print("="*80)

# Cargar dataset
df = pd.read_csv('data/02_heart_2022_clean.csv')
print(f"\n[1/7] Cargando datos...")
print(f"✓ Registros originales: {len(df):,}")

# Filtrar registros válidos
df_valido = df[df['HasCoronaryDisease'].notna()].copy()
print(f"✓ Registros válidos: {len(df_valido):,}")

# ============================================================================
# 1. PREPARAR DATOS (con mapeo explícito de ordinales)
# ============================================================================
print(f"\n[2/7] Preparando características con mapeo ordinal correcto...")

# Variable objetivo
df_valido['target'] = (df_valido['HasCoronaryDisease'] == 'Yes').astype(int)

# Características a usar
features_to_use = [
    'Sex', 'AgeCategory', 'GeneralHealth', 'UnhealthyDays', 'SleepHours',
    'HadStroke', 'HadAsthma', 'HadCOPD', 'HadDepressiveDisorder',
    'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
    'PhysicalActivities', 'SmokerStatus', 'AlcoholStatus',
    'BMI', 'CovidPos'
]

# Crear copia para transformar
X = df_valido[features_to_use].copy()
y = df_valido['target'].copy()

# --- Mapeo de variables ordinales ---
age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', 
             '45-49', '50-54', '55-59', '60-64', '65-69', 
             '70-74', '75-79', '80 or older']
age_map = {cat: i for i, cat in enumerate(age_order)}
X['AgeCategory'] = X['AgeCategory'].map(age_map)

# Salud general (de peor a mejor)
health_order = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
health_map = {cat: i for i, cat in enumerate(health_order)}
X['GeneralHealth'] = X['GeneralHealth'].map(health_map)

# Estado de fumador (de menos a más riesgo, según tu mapeo)
smoker_order = ['Never', 'Former', 'Some days', 'Every day']
smoker_map = {cat: i for i, cat in enumerate(smoker_order)}
X['SmokerStatus'] = X['SmokerStatus'].map(smoker_map)

# Estado de alcohol (de menos a más consumo)
alcohol_order = ['No', 'Occasional', 'Frequent', 'Heavy']
alcohol_map = {cat: i for i, cat in enumerate(alcohol_order)}
X['AlcoholStatus'] = X['AlcoholStatus'].map(alcohol_map)

# Sexo (binario)
X['Sex'] = (X['Sex'] == 'Female').astype(int)   # 1: Female, 0: Male

# CovidPos (binario)
X['CovidPos'] = (X['CovidPos'] == 'Yes').astype(int)

# Las variables binarias de enfermedades (HadStroke, etc.) ya son 'Yes'/'No'
for col in ['HadStroke', 'HadAsthma', 'HadCOPD', 'HadDepressiveDisorder',
            'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'PhysicalActivities']:
    X[col] = (X[col] == 'Yes').astype(int)

# Las columnas numéricas se mantienen igual
print("✓ Mapeo de variables ordinales y binarias completado")

# Eliminar filas con valores nulos (si quedan)
mask = X.isnull().any(axis=1) | y.isnull()
X = X[~mask]
y = y[~mask]
print(f"✓ Registros después de limpiar nulos: {len(X):,}")
print(f"✓ Distribución de clases - Sin enfermedad: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%) | "
      f"Con enfermedad: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")

# ============================================================================
# 2. DIVIDIR DATOS Y ENTRENAR MODELO
# ============================================================================
print(f"\n[3/7] Dividiendo datos y entrenando Random Forest...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Entrenamiento: {len(X_train):,} registros")
print(f"✓ Prueba: {len(X_test):,} registros")

# Entrenar con balanceo automático de clases
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight='balanced',   # balancea automáticamente
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Validación cruzada rápida (opcional)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"✓ Validación cruzada (AUC-ROC): media = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================================================
# 3. EVALUACIÓN
# ============================================================================
print(f"\n[4/7] Evaluando modelo...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall  # ya es la sensibilidad

print(f"✓ Accuracy: {accuracy:.4f}")
print(f"✓ Precision: {precision:.4f}")
print(f"✓ Recall (Sensibilidad): {recall:.4f}")
print(f"✓ Especificidad: {specificity:.4f}")
print(f"✓ F1-Score: {f1:.4f}")
print(f"✓ AUC-ROC: {auc_roc:.4f}")
print(f"\nMatriz de Confusión:")
print(f"  TN: {tn:,} | FP: {fp:,}")
print(f"  FN: {fn:,} | TP: {tp:,}")

# ============================================================================
# 4. IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================
print(f"\n[5/7] Características más importantes...")

feature_importance = pd.DataFrame({
    'feature': features_to_use,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 características:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

# ============================================================================
# 5. PREDICCIÓN EN TODO EL DATASET (con manejo de valores no vistos)
# ============================================================================
print(f"\n[6/7] Prediciendo en todo el dataset...")

# Usamos el mismo X original (ya limpio) para predecir
X_full = X.copy()   # ya está limpio y codificado
y_full = y.copy()

predicciones_full = model.predict(X_full)
probabilidades_full = model.predict_proba(X_full)[:, 1]

predichos_con_enfermedad = (predicciones_full == 1).sum()
predichos_sin_enfermedad = (predicciones_full == 0).sum()
porcentaje_predicho = (predichos_con_enfermedad / len(predicciones_full)) * 100

print(f"✓ Predichos con enfermedad: {predichos_con_enfermedad:,} ({porcentaje_predicho:.2f}%)")
print(f"✓ Predichos sin enfermedad: {predichos_sin_enfermedad:,}")

# ============================================================================
# 6. GUARDAR RESULTADOS (métricas + modelo entrenado)
# ============================================================================
print(f"\n[7/7] Guardando resultados...")

# Diccionario de métricas y análisis
resultados_ml = {
    'metricas_modelo': {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'sensibilidad': round(sensitivity, 4),
        'especificidad': round(specificity, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc_roc, 4),
        'cv_auc_mean': round(cv_scores.mean(), 4),
        'cv_auc_std': round(cv_scores.std(), 4),
        'registros_entrenamiento': int(len(X_train)),
        'registros_prueba': int(len(X_test)),
        'interpretacion': {
            'accuracy_explicacion': 'Porcentaje de predicciones correctas en el conjunto de prueba',
            'precision_explicacion': 'De los casos predichos como enfermos, cuántos realmente lo están',
            'recall_explicacion': 'De los casos realmente enfermos, cuántos el modelo detectó',
            'auc_roc_explicacion': 'Capacidad del modelo para distinguir entre enfermos y sanos (0.5=aleatorio, 1.0=perfecto)'
        }
    },
    'caracteristicas_importantes': feature_importance.to_dict('records'),
    'predicciones_dataset': {
        'total_predicciones': int(len(predicciones_full)),
        'predichos_con_enfermedad': int(predichos_con_enfermedad),
        'predichos_sin_enfermedad': int(predichos_sin_enfermedad),
        'porcentaje_predicho_con_enfermedad': round(porcentaje_predicho, 2),
        'nota_importante': 'El modelo predice basado en patrones en los datos.'
    },
    'matriz_confusion': {
        'verdaderos_negativos': int(tn),
        'falsos_positivos': int(fp),
        'falsos_negativos': int(fn),
        'verdaderos_positivos': int(tp),
        'interpretacion': {
            'verdaderos_negativos': 'Personas sanas correctamente identificadas',
            'falsos_positivos': 'Personas sanas predichas como enfermas',
            'falsos_negativos': 'Personas enfermas predichas como sanas (casos perdidos)',
            'verdaderos_positivos': 'Personas enfermas correctamente identificadas'
        }
    },
    'distribucion_probabilidades': {
        'promedio': round(probabilidades_full.mean(), 4),
        'mediana': round(np.median(probabilidades_full), 4),
        'desv_estandar': round(probabilidades_full.std(), 4)
    }
}

# Guardar JSON
import os
os.makedirs('data', exist_ok=True)
with open('data/06_ml_model.json', 'w', encoding='utf-8') as f:
    json.dump(resultados_ml, f, ensure_ascii=False, indent=2)

# Guardar modelo entrenado para uso futuro
with open('data/06_ml_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"✓ Resultados guardados en: data/06_ml_model.json")
print(f"✓ Modelo guardado en: data/06_ml_model.pkl")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("MODELO COMPLETADO EXITOSAMENTE")
print("="*80)
print(f"""
Métricas del Modelo:
   • Accuracy:      {accuracy:.4f}
   • Precision:     {precision:.4f}
   • Recall (Sens.):{recall:.4f}
   • Especificidad: {specificity:.4f}
   • F1-Score:      {f1:.4f}
   • AUC-ROC:       {auc_roc:.4f}  (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})

Predicciones en el Dataset:
   • Total predicciones: {len(predicciones_full):,}
   • Predichos con enfermedad: {predichos_con_enfermedad:,} ({porcentaje_predicho:.2f}%)
   • Predichos sin enfermedad: {predichos_sin_enfermedad:,}

Archivos generados:
   • data/06_ml_model.json  (métricas e informes)
   • data/06_ml_model.pkl   (modelo entrenado)
""")
print("="*80)