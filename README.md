# CardioVista Analytics

Análisis profundo de enfermedad coronaria con visualizaciones impresionantes basado en datos CDC BRFSS 2022.

## 📊 Descripción del Proyecto

CardioVista es un análisis completo de factores de riesgo de enfermedad coronaria que incluye:

- **Análisis exploratorio** de 445,11031 registros
- **Identificación de factores de riesgo** con riesgo relativo
- **Modelo ML** (XGBoost vs RandomForest)
- **Detección de personas en riesgo oculto** 
- **Visualizaciones interactivas** con Plotly

## 🎯 Hallazgos Principales

### Factores de Riesgo Críticos

| Factor | Riesgo Relativo |
|--------|-------------|
| **ACV** | 28% |
| **Enfermedad Renal** | 19% |
| **EPOC** | 17% |
| **Diabetes** |14% |
| **Artritis** | 7% |

### Edad Crítica

- **45-49 años:** 3.76% de prevalencia
- **50-54 años:** 5.45% (+1.69% en 5 años)
- **80+ años:** 23.28% (casi 1 de cada 4)

**Conclusión:** El riesgo se duplica entre 50-60 años.



## 🤖 Modelo ML

### Comparación: RandomForest vs XGBoost

| Métrica   | RandomForest   | XGBoost   | Mejora   |
|:----------|:---------------|:----------|:---------|
| Recall    | 64.83%         | 80.34%    | +15.51%  |
| Precision | 26.70%         | 22.82%    | -3.88%   |
| F1-Score  | 37.83%         | 35.54%    | -2.29%   |
| AUC-ROC   | 83.27%         | 84.47%    | +1.20%   |

### Interpretación

**XGBoost es mejor para este caso porque:**
- Detecta **80.34%** de los enfermos (vs 64.83% en RandomForest)
- Ideal para screening inicial de poblaciones
- Sacrifica precisión por recall (mejor perder falsos positivos que enfermos)

### Top 5 Características Importantes

1. **Edad** (29.07%)
2. **IMC** (11.33%)
3. **Salud General** (9.77%)
4. **Artritis** (6.48%)
5. **Diabetes** (6.35%)

## 📁 Estructura del Proyecto

```
visualization-heart-diseases/
visualization-heart-diseases/
├── css/
│   └── style.css
├── data/
│       ├── 03_basic_analysis.json
│       ├── 04_risk_analysis.json
│       ├── 05_advanced_risks.json
│       ├── 06_ml_model.json     
│       ├── 08_ml_insights.json 
│       └── 10_xgboost_model.json
├── js/
│   └── main.js                  
├── python/                   
│   ├── 01_convert_parquet.py
│   ├── 02_cleaning.py
│   ├── 03_basic_analysis.py
│   ├── 04_risk_analysis.py
│   ├── 05_advanced_risks.py
│   ├── 06_ml_model.py
│   ├── 07_ml_insights.py      
│   └── 10_xgboost_model.py
├── index.html
├── README.md
├── .gitignore
├── LICENSE
└── requirements.txt
```

## 🚀 Cómo Usar

### 1. Ejecutar el Análisis

```bash
# Instalar dependencias
pip install pandas numpy scikit-learn xgboost

# Ejecutar scripts en orden
python python/01_convert_parquet.py
python python/02_cleaning.py
python python/03_basic_analysis.py
python python/04_risk_analysis.py
python python/05_advanced_risks.py
python python/06_ml_model.py
python python/07_ml_insights.py
```

### 2. Ver el Dashboard

Abre `index.html` en tu navegador para ver las visualizaciones interactivas.

```bash
# Opción 1: Abrir directamente
open index.html

# Opción 2: Usar un servidor local
python -m http.server 8000
# Luego abre http://localhost:8000
```



## 📊 Visualizaciones Incluidas

### Sección: Datos
* **Resumen Estadístico:** Métricas clave sobre el conjunto de datos, registros analizados y cobertura geográfica.
* **Metodología:** Detalle sobre el proceso de limpieza, estandarización de variables y tratamiento de valores atípicos.

### Sección: Análisis Geográfico y Demográfico
* **Mapa de Prevalencia:** Visualización interactiva de la distribución de enfermedades coronarias a nivel estatal.
* **Distribución:** Comparativa demográfica por sexo y raza mediante gráficos de waffle y análisis de prevalencia segmentada.

### Sección: Factores de Riesgo
1. **Timeline de Edad**
   * **Descripción:** Visualización del crecimiento exponencial del riesgo cardiovascular a medida que aumenta la edad.
   * **Objetivo:** Identificar los grupos etarios críticos donde la prevención debe intensificarse.

2. **Gráfico de Burbujas**
   * **Descripción:** Representación donde el **tamaño** de la burbuja equivale al volumen de población y el **color** indica el nivel de riesgo.
   * **Objetivo:** Visualizar simultáneamente la prevalencia y la gravedad en diferentes estratos.

3. **Comorbilidades Críticas**
   * **Descripción:** Análisis de las combinaciones de patologías con mayor impacto negativo en la salud coronaria.
   * **Objetivo:** Detectar interacciones de enfermedades que multiplican el riesgo base.

---

## 🤖 Sección: Modelo ML

1. **Comparación de Métricas**
   * **Descripción:** Cuadro comparativo entre *Random Forest* y *XGBoost*.
   * **Objetivo:** Evaluar la mejora en rendimiento (Recall, Precision, F1-Score, AUC-ROC) tras el ajuste del modelo.

2. **Matriz de Confusión**
   * **Descripción:** Visualización de TP (Verdaderos Positivos), FN (Falsos Negativos), TN (Verdaderos Negativos) y FP (Falsos Positivos).
   * **Objetivo:** Entender los errores del modelo y el equilibrio entre sensibilidad y especificidad.

3. **Curva ROC**
   * **Descripción:** Representación gráfica de la tasa de verdaderos positivos frente a la tasa de falsos positivos.
   * **Objetivo:** Medir la capacidad de discriminación del modelo a diferentes umbrales de decisión.

4. **Feature Importance**
   * **Descripción:** Gráfico de barras que muestra las 10 características (variables) con mayor peso en la predicción final.
   * **Objetivo:** Identificar qué factores clínicos y demográficos son los determinantes principales para el modelo.

---

## 🚀 Tecnologías Utilizadas
* **Python:** Procesamiento y modelado (Scikit-learn, XGBoost, Pandas).
* **Frontend:** HTML5, CSS3, JavaScript.
* **Visualización:** Plotly.js, D3.js, TopoJSON.



## 🔍 Explicaciones Clave

### ¿Qué significa Recall 80.34%?

El modelo detecta 80 de cada 100 personas enfermas. Los 20 restantes son "falsos negativos" (personas enfermas que el modelo no detectó).

### ¿Qué significa Precision 22.82%?

De cada 100 personas que el modelo predice como enfermas, solo 23 realmente lo están. Los 77 restantes son "falsos positivos".

### ¿Por qué XGBoost es mejor?

Para detección de enfermedades, es mejor tener más falsos positivos (personas sanas predichas como enfermas) que falsos negativos (personas enfermas no detectadas). XGBoost optimiza para esto.



## 🛠️ Tecnologías Utilizadas

- **Python 3.11**
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Scikit-learn** - RandomForest
- **XGBoost** - Gradient Boosting
- **Plotly** - Visualizaciones interactivas
- **HTML/CSS/JavaScript** - Dashboard


## 📝 Notas Importantes

1. **El modelo NO es diagnóstico:** Solo es una herramienta de screening. Requiere confirmación clínica.
2. **Desbalance de clases:** Solo 9% tiene enfermedad. El modelo está optimizado para detectar la minoría.
3. **Recall vs Precision:** Se priorizó Recall (detectar más enfermos) sobre Precision.

## 🔗 Fuentes
- [CDC BRFSS 2022](https://www.cdc.gov/brfss/annual_data/annual_2022.html/)
- [CDC BRFSS 2022 BASE](https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip)
- [GITHUB BASE](https://github.com/kamilpytlak/data-science-projects/blob/main/heart-disease-prediction/2022/notebooks/data_processing.ipynb)



## 📄 Licencia

MIT License - Libre para usar, modificar y distribuir.



## 👨‍💻 Autor

- Universidad: UOC
- Master: En Ciencia de Datos
- Materia: Visualización de Datos
- Autor: Eduardo Daniel Mayoral Chavando

---

**Última actualización:** Mayo 2026
