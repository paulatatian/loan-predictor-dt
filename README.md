# 🏦 Predictor de Aprobación de Préstamos

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://TU-URL-AQUI.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Descripción

Aplicación web de Machine Learning que predice si una solicitud de préstamo bancario
será **aprobada o rechazada**, basándose en características del solicitante como ingresos,
historial crediticio, nivel educativo y otros factores financieros.

Esta herramienta simula el proceso de evaluación crediticia automatizada que los bancos
utilizan para agilizar la toma de decisiones sobre préstamos.

---

## 🚀 Demostración

> 🔗 **[Ver aplicación en vivo](https://TU-URL-AQUI.streamlit.app)**

![App Screenshot](docs/screenshot.png)

---

## 🤖 Algoritmo utilizado

**Árbol de Decisión (Decision Tree Classifier)**

Un Árbol de Decisión es un modelo que aprende reglas de decisión a partir de los datos,
similar a un diagrama de flujo. Cada nodo del árbol hace una pregunta sobre una variable
(por ejemplo: "¿tiene buen historial crediticio?") y las ramas llevan hacia la decisión final.

**¿Por qué es apropiado para este problema?**
- **Interpretable:** Los bancos deben poder explicar por qué se aprueba o rechaza un préstamo (regulaciones financieras)
- **Maneja variables mixtas:** Funciona bien con datos numéricos y categóricos sin normalización
- **Bajo costo computacional:** Predicciones en tiempo real sin demoras
- **Transparent:** Se puede visualizar el árbol completo de decisiones

### Métricas de desempeño

| Métrica | Valor |
|---------|-------|
| Accuracy | ~82% |
| Precision | ~84% |
| Recall | ~87% |
| F1-Score | ~85% |

> Los valores exactos dependen del split de datos. Ver notebook para resultados actualizados.

---

## 📊 Dataset

- **Fuente:** [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Tamaño:** 614 registros reales
- **Features utilizadas:** 11 variables

| Variable | Descripción | Tipo |
|----------|-------------|------|
| Gender | Género del solicitante | Categórica |
| Married | Estado civil | Categórica |
| Dependents | Número de dependientes | Numérica |
| Education | Nivel educativo | Categórica |
| Self_Employed | ¿Trabaja por cuenta propia? | Categórica |
| ApplicantIncome | Ingresos del solicitante | Numérica |
| CoapplicantIncome | Ingresos del co-solicitante | Numérica |
| LoanAmount | Monto del préstamo (miles) | Numérica |
| Loan_Amount_Term | Plazo en meses | Numérica |
| Credit_History | Historial crediticio (0/1) | Numérica |
| Property_Area | Área de la propiedad | Categórica |

**Variable objetivo:** `Loan_Status` (Y = Aprobado, N = Rechazado)

---

## 🛠️ Instalación local

### Requisitos
- Python 3.11+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU-USUARIO/loan-predictor-dt.git
cd loan-predictor-dt

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Obtener el dataset
# Descargar train.csv desde Kaggle y guardarlo como:
# data/raw/loan_data.csv

# 4. Entrenar el modelo
jupyter notebook notebooks/01_training.ipynb
# (ejecutar todas las celdas)

# 5. Ejecutar la aplicación
streamlit run app/app.py
```

La app abrirá automáticamente en `http://localhost:8501`

---

## 🗂️ Estructura del proyecto

```
loan-predictor-dt/
│
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias Python
├── .gitignore                  # Archivos ignorados por Git
│
├── data/
│   ├── raw/                    # Dataset original (loan_data.csv)
│   └── processed/              # Datos procesados
│
├── notebooks/
│   └── 01_training.ipynb       # Notebook de entrenamiento completo
│
├── models/
│   ├── loan_model.pkl          # Modelo entrenado serializado
│   ├── feature_names.pkl       # Nombres de features
│   └── label_mapeos.pkl        # Mapeos de variables categóricas
│
├── app/
│   └── app.py                  # Aplicación Streamlit
│
└── docs/
    ├── arbol_decision.png      # Visualización del árbol
    ├── matriz_confusion.png    # Matriz de confusión
    ├── feature_importance.png  # Importancia de variables
    └── presentacion.pdf        # Presentación final
```

---

## 📱 Uso de la aplicación

1. Ingresa los datos del solicitante en el formulario de la izquierda:
   - Datos personales (género, estado civil, educación)
   - Datos financieros (ingresos, monto del préstamo, plazo)
   - Historial crediticio y área de la propiedad
2. Presiona el botón **"Analizar Solicitud"**
3. El modelo muestra:
   - ✅ Aprobado o ❌ Rechazado
   - Probabilidad de cada resultado
   - Los factores más influyentes en la decisión

---

## ☁️ Despliegue

La aplicación está desplegada en **Streamlit Cloud** (gratuito):

1. Subir código a GitHub (repositorio público)
2. Ir a [streamlit.io/cloud](https://streamlit.io/cloud)
3. Conectar el repositorio
4. Seleccionar `app/app.py` como archivo principal
5. Hacer clic en **Deploy**

---

## 👥 Autores

| Nombre | Rol |
|--------|-----|
| [Nombre 1] | Entrenamiento del modelo y análisis de datos |
| [Nombre 2] | Desarrollo de la aplicación web |
| [Nombre 3] | Despliegue en la nube y documentación |

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE) para detalles.

---

## 📚 Referencias

- Kaggle Dataset: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- scikit-learn Decision Trees: https://scikit-learn.org/stable/modules/tree.html
- Streamlit Documentation: https://docs.streamlit.io
- Streamlit Cloud Deployment: https://docs.streamlit.io/streamlit-community-cloud
