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

## ❗ Problemática y Justificación

En Colombia, el acceso a la educación superior sigue siendo uno de los principales retos socioeconómicos del país. Según datos del ICETEX, una proporción significativa de estudiantes que solicitan crédito educativo desconoce cuál modalidad de financiación se ajusta mejor a su perfil, lo que genera:

- **Solicitudes mal dirigidas:** Los solicitantes invierten tiempo y esfuerzo en tramitar modalidades que no corresponden a su condición socioeconómica o nivel de formación, aumentando las tasas de rechazo.
- **Deserción por desinformación:** Estudiantes que podrían acceder a financiación de sostenimiento no lo solicitan por no conocer que aplican a ella, lo que en muchos casos los lleva a abandonar sus estudios.
- **Inequidad en el acceso:** Las poblaciones más vulnerables (estratos 1–2, municipios de categorías bajas, regiones apartadas) son precisamente las que menos información tienen sobre las opciones disponibles, perpetuando la brecha educativa.
- **Carga operativa en ICETEX:** El volumen de solicitudes con información incompleta o modalidad incorrecta genera reprocesos administrativos que ralentizan la asignación de créditos.

Esta aplicación responde directamente a esa necesidad: **orientar al estudiante antes de que presente su solicitud**, indicándole con base en su perfil real cuál modalidad tiene mayor probabilidad de corresponderle. Esto no solo mejora la experiencia del solicitante, sino que potencialmente reduce los tiempos de trámite y aumenta las tasas de aprobación.

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

## 🗃️ Fuente de Datos

- **Origen:** [ICETEX — Datos Abiertos Colombia](https://www.datos.gov.co)
- **Dataset:** Créditos ICETEX Otorgados
- **Registros:** 109.139 créditos reales
- **País:** 🇨🇴 Colombia

> ⚠️ El dataset original no está incluido en este repositorio. Descargarlo desde el portal de Datos Abiertos del Gobierno de Colombia.

---

## ✅ Conclusión

Este proyecto demuestra cómo la inteligencia artificial puede aplicarse a problemas concretos de política pública y equidad educativa. A través de un modelo de árbol de decisión entrenado con más de 109.000 registros reales, se logró construir un sistema de apoyo a la decisión que traduce variables del perfil del estudiante en una predicción interpretable y accionable sobre la modalidad de crédito más probable.

Más allá del ejercicio académico, la aplicación evidencia el potencial de los modelos de clasificación para reducir asimetrías de información en procesos críticos como el acceso al crédito educativo. Un estudiante mejor orientado tiene más probabilidades de completar su solicitud correctamente, de acceder al financiamiento que le corresponde y, en última instancia, de continuar con su formación profesional.

Como trabajo futuro, se identifican oportunidades para enriquecer el modelo con variables adicionales (como puntaje Saber 11 o situación laboral del núcleo familiar), explorar algoritmos de mayor capacidad predictiva como Random Forest o Gradient Boosting, e integrar el sistema con fuentes de datos en tiempo real del portal de Datos Abiertos de Colombia, con el fin de mantener el modelo actualizado y ampliar su utilidad en escenarios reales de orientación estudiantil.

---

## 👩‍💻 Autores

Desarrollado como parte de la asignatura **Inteligencia Artificial I** — Actividad 3.

---

## 📄 Licencia

Este proyecto es de uso académico. Los datos pertenecen al sistema de datos abiertos del Gobierno de Colombia.
