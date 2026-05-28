# 🎓 Predictor de Modalidad de Crédito ICETEX

> **Inteligencia Artificial I — Actividad 3**  
> Modelo de clasificación con Árbol de Decisión aplicado a datos reales de créditos ICETEX Colombia

---

## 📌 Descripción

Aplicación web interactiva que predice la **modalidad de crédito ICETEX** que recibiría un estudiante colombiano según su perfil socioeconómico y académico. El modelo fue entrenado con **109.139 registros reales** del sistema de créditos educativos de Colombia.

Las modalidades que puede predecir son:
- 📗 **Matrícula** — Financiación del valor de la matrícula universitaria
- 📘 **Sostenimiento** — Apoyo para gastos de manutención del estudiante
- 📙 **Otra modalidad** — Otros tipos de crédito disponibles en ICETEX

---

## 🧠 Algoritmo

| Parámetro | Valor |
|---|---|
| Algoritmo | Árbol de Decisión (`DecisionTreeClassifier`) |
| Librería | `scikit-learn` |
| Registros de entrenamiento | 109.139 |
| Variables de entrada | 8 features |
| Variable objetivo | `MODALIDAD DEL CRÉDITO` |

---

## 📂 Estructura del Proyecto

```
proyecto-icetex/
│
├── app/
│   └── app.py                  # Aplicación Streamlit (UI principal)
│
├── models/
│   ├── icetex_model.pkl        # Modelo entrenado serializado
│   ├── label_encoders.pkl      # Encoders para variables categóricas
│   └── clases.pkl              # Clases de la variable objetivo
│
├── notebooks/
│   └── 01_training_icetex.ipynb  # Notebook de entrenamiento y EDA
│
├── data/                       # Dataset ICETEX (no incluido en el repo)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Variables de Entrada (Features)

| # | Variable | Tipo | Descripción |
|---|---|---|---|
| 1 | `SEXO AL NACER` | Categórica | Sexo biológico del solicitante |
| 2 | `ESTRATO SOCIOECONÓMICO` | Numérica (1–6) | Estrato del hogar del estudiante |
| 3 | `DEPARTAMENTO DE ORIGEN` | Categórica | Departamento colombiano de origen |
| 4 | `CATEGORÍA DEL MUNICIPIO DE ORIGEN` | Categórica | Categoría del municipio según clasificación oficial |
| 5 | `SECTOR IES` | Categórica | Sector de la institución educativa (Oficial / Privada) |
| 6 | `NIVEL DE FORMACIÓN` | Categórica | Nivel académico (Técnico, Tecnólogo, Universitario, Posgrado, etc.) |
| 7 | `MODALIDAD DE LÍNEA` | Categórica | Línea del crédito (Pregrado, Posgrado País, Exterior) |
| 8 | `RANGO DEL VALOR TOTAL DESEMBOLSADO` | Categórica | Decil del valor desembolsado (I = menor, X = mayor) |

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/predictor-icetex.git
cd predictor-icetex
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Entrenar el modelo

Abrir y ejecutar todas las celdas del notebook:

```bash
jupyter notebook notebooks/01_training_icetex.ipynb
```

Esto generará los archivos `.pkl` en la carpeta `models/`.

### 4. Lanzar la aplicación

```bash
streamlit run app/app.py
```

La aplicación estará disponible en `http://localhost:8501`

---

## 📦 Dependencias

```
streamlit
scikit-learn
pandas
numpy
joblib
```

Instalar con:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Vista de la Aplicación

La interfaz cuenta con:

- **Formulario de entrada** con los 8 campos del perfil del estudiante
- **Resultado de predicción** con tarjeta visual codificada por color
- **Gráfico de probabilidades** por cada modalidad posible
- **Variables más influyentes** del modelo para la predicción actual
- **Panel lateral** con información del modelo y el dataset

---

## 🗃️ Fuente de Datos

- **Origen:** [ICETEX — Datos Abiertos Colombia](https://www.datos.gov.co)
- **Dataset:** Créditos ICETEX Otorgados
- **Registros:** 109.139 créditos reales
- **País:** 🇨🇴 Colombia

> ⚠️ El dataset original no está incluido en este repositorio. Descargarlo desde el portal de Datos Abiertos del Gobierno de Colombia.

---

## 📊 Pipeline de Entrenamiento

```
Datos crudos (CSV)
    │
    ▼
Limpieza y preprocesamiento
    │  - Eliminación de nulos
    │  - LabelEncoding de variables categóricas
    ▼
División train/test (80/20)
    │
    ▼
Entrenamiento DecisionTreeClassifier
    │
    ▼
Evaluación del modelo
    │  - Accuracy
    │  - Matriz de confusión
    │  - Reporte de clasificación
    ▼
Serialización con joblib
    │  → icetex_model.pkl
    │  → label_encoders.pkl
    │  → clases.pkl
```

---

## 👩‍💻 Autores

Desarrollado como parte de la asignatura **Inteligencia Artificial I** — Actividad 3.

---

## 📄 Licencia

Este proyecto es de uso académico. Los datos pertenecen al sistema de datos abiertos del Gobierno de Colombia.
