"""
Análisis profundo del modelo ML: Capacidad predictiva y personas no diagnosticadas
Genera explicaciones sobre qué puede predecir el modelo y quién está en riesgo
Incluye las 10 variables más importantes según Random Forest
"""

import pandas as pd
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("="*80)
print("ANÁLISIS PROFUNDO DEL MODELO ML")
print("="*80)

# Cargar datos
df = pd.read_csv('data/02_heart_2022_clean.csv')
df_valido = df[df['HasCoronaryDisease'].notna()].copy()

print(f"\n[1/7] Preparando datos...")

# Preparar datos
df_valido['target'] = (df_valido['HasCoronaryDisease'] == 'Yes').astype(int)

features_to_use = [
    'Sex', 'AgeCategory', 'GeneralHealth', 'UnhealthyDays', 'SleepHours',
    'HadStroke', 'HadAsthma', 'HadCOPD', 'HadDepressiveDisorder',
    'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
    'PhysicalActivities', 'SmokerStatus', 'AlcoholStatus',
    'BMI', 'CovidPos'
]

X = df_valido[features_to_use].copy()
y = df_valido['target'].copy()

mask = X.isnull().any(axis=1) | y.isnull()
X = X[~mask]
y = y[~mask]

# Codificar
label_encoders = {}
X_encoded = X.copy()

for col in X_encoded.columns:
    if not pd.api.types.is_numeric_dtype(X_encoded[col]):
        le = LabelEncoder()
        encoded_values = le.fit_transform(X_encoded[col].astype(str))
        X_encoded[col] = encoded_values.astype(float)
        label_encoders[col] = le
    else:
        X_encoded[col] = X_encoded[col].astype(float)

# Entrenar modelo
print(f"\n[2/7] Entrenando modelo...")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

class_weight = {0: 1, 1: (len(y_train) - y_train.sum()) / y_train.sum()}

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ============================================================================
# IMPORTANCIA DE CARACTERÍSTICAS (TOP 10)
# ============================================================================
print(f"\n[3/7] Calculando importancia de características...")

feature_importance = pd.DataFrame({
    'feature': features_to_use,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

top_10_features = feature_importance.head(10).to_dict('records')

print("\nTOP 10 variables más importantes:")
for i, row in enumerate(feature_importance.head(10).itertuples(), 1):
    print(f"  {i}. {row.feature:25s}: {row.importance:.4f}")

# ============================================================================
# EVALUACIÓN DEL MODELO
# ============================================================================
print(f"\n[4/7] Evaluando modelo...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

# ============================================================================
# ANÁLISIS DE CAPACIDAD PREDICTIVA
# ============================================================================
print(f"\n[5/7] Analizando capacidad predictiva...")

capacidad_predictiva = {
    'accuracy': {
        'valor': round(accuracy, 4),
        'porcentaje': round(accuracy * 100, 2),
        'explicacion': f'El modelo acierta en {round(accuracy*100, 1)}% de los casos. De cada 100 predicciones, {round(accuracy*100, 1)} son correctas.',
        'interpretacion': 'Buena precisión general, pero no es suficiente para confiar únicamente en el modelo.'
    },
    'recall': {
        'valor': round(recall, 4),
        'porcentaje': round(recall * 100, 2),
        'explicacion': f'El modelo detecta {round(recall*100, 1)}% de los casos reales de enfermedad. De cada 100 enfermos, identifica {round(recall*100, 1)}.',
        'interpretacion': f'Esto significa que {round((1-recall)*100, 1)}% de los enfermos NO son detectados por el modelo. Son casos perdidos.',
        'casos_perdidos': int(fn)
    },
    'precision': {
        'valor': round(precision, 4),
        'porcentaje': round(precision * 100, 2),
        'explicacion': f'De los casos que el modelo predice como enfermos, {round(precision*100, 1)}% realmente lo están.',
        'interpretacion': 'Hay falsos positivos (personas sanas predichas como enfermas), pero son útiles para alertar sobre riesgo.'
    },
    'auc_roc': {
        'valor': round(auc_roc, 4),
        'porcentaje': round(auc_roc * 100, 2),
        'explicacion': f'El modelo tiene {round(auc_roc*100, 1)}% de capacidad para distinguir entre enfermos y sanos.',
        'interpretacion': 'Excelente discriminación. El modelo es muy bueno identificando quién está en riesgo vs quién no.'
    }
}

# ============================================================================
# ANÁLISIS DE MATRIZ DE CONFUSIÓN
# ============================================================================
print(f"\n[6/7] Analizando matriz de confusión...")

matriz_analisis = {
    'verdaderos_positivos': {
        'cantidad': int(tp),
        'explicacion': f'{tp:,} personas enfermas fueron correctamente identificadas como enfermas.'
    },
    'falsos_negativos': {
        'cantidad': int(fn),
        'explicacion': f'{fn:,} personas enfermas fueron predichas como sanas. ESTOS SON LOS CASOS PERDIDOS - personas que tienen enfermedad pero el modelo no las detectó.',
        'riesgo': f'Estas {fn:,} personas necesitan pruebas adicionales porque podrían tener enfermedad sin saberlo.'
    },
    'verdaderos_negativos': {
        'cantidad': int(tn),
        'explicacion': f'{tn:,} personas sanas fueron correctamente identificadas como sanas.'
    },
    'falsos_positivos': {
        'cantidad': int(fp),
        'explicacion': f'{fp:,} personas sanas fueron predichas como enfermas. Estos son falsos positivos.',
        'utilidad': 'Aunque son falsos positivos, son útiles para alertar a personas aparentemente sanas pero con factores de riesgo.'
    }
}

# ============================================================================
# PREDICCIONES EN TODO EL DATASET
# ============================================================================
print(f"\n[7/7] Analizando predicciones en todo el dataset...")

# Preparar todo el dataset
X_full = df_valido[features_to_use].copy()
mask_full = X_full.isnull().any(axis=1)
X_full_clean = X_full[~mask_full].copy()

X_full_encoded = X_full_clean.copy()
for col in X_full_encoded.columns:
    if not pd.api.types.is_numeric_dtype(X_full_encoded[col]):
        encoded_values = label_encoders[col].transform(X_full_encoded[col].astype(str))
        X_full_encoded[col] = encoded_values.astype(float)
    else:
        X_full_encoded[col] = X_full_encoded[col].astype(float)

predicciones_full = model.predict(X_full_encoded)
probabilidades_full = model.predict_proba(X_full_encoded)[:, 1]

predichos_con_enfermedad = (predicciones_full == 1).sum()
predichos_sin_enfermedad = (predicciones_full == 0).sum()
porcentaje_predicho = (predichos_con_enfermedad / len(predicciones_full)) * 100

# Análisis de probabilidades
prob_mean = probabilidades_full.mean()
prob_median = np.median(probabilidades_full)
prob_std = probabilidades_full.std()

# Personas con alta probabilidad pero sin diagnóstico
df_full_pred = df_valido[~df_valido.index.isin(df_valido[mask].index)].copy()
df_full_pred['pred_proba'] = probabilidades_full
df_full_pred['pred_class'] = predicciones_full

# Alto riesgo sin diagnóstico
alto_riesgo_sin_diag = df_full_pred[
    (df_full_pred['HasCoronaryDisease'] == 'No') & 
    (df_full_pred['pred_proba'] > 0.5)
]

# Bajo riesgo con diagnóstico
bajo_riesgo_con_diag = df_full_pred[
    (df_full_pred['HasCoronaryDisease'] == 'Yes') & 
    (df_full_pred['pred_proba'] < 0.3)
]

predicciones_analisis = {
    'total_predicciones': int(len(predicciones_full)),
    'predichos_con_enfermedad': int(predichos_con_enfermedad),
    'predichos_sin_enfermedad': int(predichos_sin_enfermedad),
    'porcentaje_predicho': round(porcentaje_predicho, 2),
    'explicacion': f'El modelo predice que {predichos_con_enfermedad:,} personas ({round(porcentaje_predicho, 1)}%) están en riesgo de enfermedad coronaria.',
    'distribucion_probabilidades': {
        'promedio': round(prob_mean, 4),
        'mediana': round(prob_median, 4),
        'desv_estandar': round(prob_std, 4)
    },
    'hallazgos_clave': {
        'alto_riesgo_sin_diagnostico': {
            'cantidad': int(len(alto_riesgo_sin_diag)),
            'porcentaje': round(len(alto_riesgo_sin_diag) / len(df_full_pred) * 100, 2),
            'explicacion': f'{len(alto_riesgo_sin_diag):,} personas NO tienen diagnóstico de enfermedad coronaria, pero el modelo les da >50% de probabilidad de tenerla.',
            'recomendacion': 'Estas personas deberían someterse a pruebas de detección (ECG, ecocardiograma, pruebas de estrés).'
        },
        'bajo_riesgo_con_diagnostico': {
            'cantidad': int(len(bajo_riesgo_con_diag)),
            'porcentaje': round(len(bajo_riesgo_con_diag) / len(df_full_pred) * 100, 2),
            'explicacion': f'{len(bajo_riesgo_con_diag):,} personas TIENEN diagnóstico pero el modelo les da <30% de probabilidad.',
            'interpretacion': 'Podrían ser casos diagnosticados tempranamente o con factores de riesgo menos evidentes.'
        }
    }
}

# ============================================================================
# GUARDAR ANÁLISIS (incluyendo top 10 features)
# ============================================================================
print(f"\nGuardando análisis...")

analisis_ml = {
    'caracteristicas_importantes': top_10_features,
    'capacidad_predictiva': capacidad_predictiva,
    'matriz_confusion': matriz_analisis,
    'predicciones_dataset': predicciones_analisis,
    'resumen_ejecutivo': {
        'pregunta_clave': '¿Cuántas personas podrían tener enfermedad coronaria sin saberlo?',
        'respuesta': f'Según el modelo, aproximadamente {len(alto_riesgo_sin_diag):,} personas ({round(len(alto_riesgo_sin_diag)/len(df_full_pred)*100, 2)}%) tienen alta probabilidad de enfermedad pero no tienen diagnóstico.',
        'capacidad_del_modelo': f'El modelo puede identificar correctamente {round(recall*100, 1)}% de los casos reales, pero pierde {round((1-recall)*100, 1)}% (casos perdidos).',
        'recomendacion': 'El modelo es útil para identificar poblaciones en riesgo, pero debe complementarse con pruebas clínicas reales.',
        'top_3_variables': {v['feature']: round(v['importance'],4) for v in top_10_features[:3]}  # resumen rápido
    }
}

import os
if not os.path.exists('data'):
    os.makedirs('data')

with open('data/08_ml_insights.json', 'w', encoding='utf-8') as f:
    json.dump(analisis_ml, f, ensure_ascii=False, indent=2)

print(f"✓ Análisis guardado en: data/08_ml_insights.json")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "="*80)
print("ANÁLISIS ML COMPLETADO")
print("="*80)

print(f"""
TOP 10 VARIABLES MÁS IMPORTANTES:
""")
for i, row in enumerate(feature_importance.head(10).itertuples(), 1):
    print(f"   {i}. {row.feature}: {row.importance:.4f}")

print(f"""
CAPACIDAD PREDICTIVA DEL MODELO:

Métricas Clave:
   • Accuracy: {round(accuracy*100, 2)}% - Acierta en {round(accuracy*100, 1)} de cada 100 predicciones
   • Recall: {round(recall*100, 2)}% - Detecta {round(recall*100, 1)} de cada 100 enfermos reales
   • Precision: {round(precision*100, 2)}% - De los predichos como enfermos, {round(precision*100, 1)}% realmente lo están
   • AUC-ROC: {round(auc_roc*100, 2)}% - Excelente capacidad de discriminación

Matriz de Confusión (en conjunto de prueba):
   • Verdaderos Positivos: {tp:,} (enfermos correctamente identificados)
   • Falsos Negativos: {fn:,} (enfermos NO detectados - CASOS PERDIDOS)
   • Verdaderos Negativos: {tn:,} (sanos correctamente identificados)
   • Falsos Positivos: {fp:,} (sanos predichos como enfermos)

Personas No Diagnosticadas en Riesgo:
   • Alto riesgo (>50% probabilidad): {len(alto_riesgo_sin_diag):,} personas
   • Estos son candidatos para pruebas de detección

Limitaciones del Modelo:
   • Pierde {round((1-recall)*100, 1)}% de los casos reales (falsos negativos)
   • No debe usarse como diagnóstico definitivo
   • Debe complementarse con evaluación clínica

Archivo generado: data/08_ml_insights.json
""")

print("="*80)