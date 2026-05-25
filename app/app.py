"""
Aplicación web para predicción de aprobación de préstamos
Algoritmo: Árbol de Decisión
Inteligencia Artificial I - Actividad 3
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Préstamos",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Estilos CSS personalizados
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Tarjeta de resultado */
    .resultado-aprobado {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 1.6em;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(56, 239, 125, 0.3);
        margin: 20px 0;
        animation: fadeIn 0.5s ease;
    }

    .resultado-rechazado {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 1.6em;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(231, 76, 60, 0.3);
        margin: 20px 0;
        animation: fadeIn 0.5s ease;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }

    .metric-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Estilo de botón */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 0;
        font-size: 1.1em;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    h1, h2, h3 { color: white !important; }
    label { color: #ddd !important; }
    p { color: #ccc; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cargar el modelo
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """Carga el modelo desde el archivo .pkl. Solo se ejecuta una vez."""
    ruta_modelo = os.path.join(os.path.dirname(__file__), '..', 'models', 'loan_model.pkl')
    try:
        modelo = joblib.load(ruta_modelo)
        return modelo
    except FileNotFoundError:
        return None

modelo = cargar_modelo()


# ─────────────────────────────────────────────
# Encabezado principal
# ─────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    st.markdown("<div style='font-size:4em; text-align:center; padding-top:10px'>🏦</div>",
                unsafe_allow_html=True)
with col_titulo:
    st.markdown("<h1 style='margin-bottom:0'>Predictor de Aprobación de Préstamos</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa; font-size:1.05em'>Powered by Árbol de Decisión · Inteligencia Artificial I</p>",
                unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# Verificar si el modelo está disponible
# ─────────────────────────────────────────────
if modelo is None:
    st.error("""
    ⚠️ **Modelo no encontrado**

    Para usar esta aplicación debes primero entrenar el modelo:
    1. Abre el notebook `notebooks/01_training.ipynb`
    2. Ejecuta todas las celdas
    3. El modelo se guardará automáticamente en `models/loan_model.pkl`
    4. Recarga esta página
    """)
    st.stop()

# ─────────────────────────────────────────────
# Layout: formulario izquierda, resultado derecha
# ─────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:
    st.markdown("### 📋 Información del Solicitante")
    st.markdown("<p>Completa los datos para obtener la predicción del modelo.</p>",
                unsafe_allow_html=True)

    # Sección 1: Datos personales
    st.markdown("**👤 Datos Personales**")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Género", ["Male", "Female"],
                              help="Género del solicitante principal")
        education = st.selectbox("Nivel educativo",
                                 ["Graduate", "Not Graduate"],
                                 help="Graduate = universitario, Not Graduate = no universitario")
    with c2:
        married = st.selectbox("Estado civil", ["Yes", "No"],
                               help="¿El solicitante está casado?")
        self_employed = st.selectbox("¿Trabaja por cuenta propia?",
                                     ["No", "Yes"])

    dependents = st.select_slider(
        "Número de dependientes",
        options=["0", "1", "2", "3+"],
        help="Personas que dependen económicamente del solicitante"
    )

    st.markdown("---")

    # Sección 2: Datos financieros
    st.markdown("**💰 Datos Financieros**")
    c3, c4 = st.columns(2)
    with c3:
        applicant_income = st.number_input(
            "Ingresos mensuales (USD)",
            min_value=0, max_value=100000, value=4000, step=500,
            help="Ingresos mensuales del solicitante principal"
        )
        loan_amount = st.number_input(
            "Monto del préstamo (miles USD)",
            min_value=1, max_value=700, value=120, step=10,
            help="Monto del préstamo solicitado en miles de dólares"
        )
    with c4:
        coapplicant_income = st.number_input(
            "Ingresos co-solicitante (USD)",
            min_value=0, max_value=50000, value=0, step=500,
            help="Ingresos del cónyuge u otro co-solicitante (0 si no aplica)"
        )
        loan_term = st.selectbox(
            "Plazo del préstamo (meses)",
            options=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
            index=9,
            help="Número de meses para pagar el préstamo"
        )

    st.markdown("---")

    # Sección 3: Historial y propiedad
    st.markdown("**🏠 Historial y Propiedad**")
    c5, c6 = st.columns(2)
    with c5:
        credit_history = st.radio(
            "Historial crediticio",
            options=[1, 0],
            format_func=lambda x: "✅ Bueno (pagos al día)" if x == 1 else "❌ Malo (deudas/mora)",
            help="¿El solicitante tiene buen historial de pagos?"
        )
    with c6:
        property_area = st.selectbox(
            "Área de la propiedad",
            ["Urban", "Semiurban", "Rural"],
            help="Ubicación de la propiedad a financiar"
        )

    st.markdown("")
    predecir = st.button("🔍 Analizar Solicitud", use_container_width=True)


# ─────────────────────────────────────────────
# Resultado de la predicción
# ─────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Resultado del Análisis")

    if not predecir:
        # Estado inicial
        st.markdown("""
        <div class='info-card' style='text-align:center; padding: 50px 20px;'>
            <div style='font-size:3em'>🤖</div>
            <h3 style='color:#aaa !important'>Esperando datos...</h3>
            <p style='color:#888'>Completa el formulario de la izquierda<br>y presiona <b>Analizar Solicitud</b></p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Preprocesar inputs ──────────────────
        mapeo_gender   = {"Male": 1, "Female": 0}
        mapeo_married  = {"Yes": 1, "No": 0}
        mapeo_dep      = {"0": 0, "1": 1, "2": 2, "3+": 3}
        mapeo_edu      = {"Graduate": 0, "Not Graduate": 1}
        mapeo_self     = {"Yes": 1, "No": 0}
        mapeo_area     = {"Rural": 0, "Semiurban": 1, "Urban": 2}

        input_data = pd.DataFrame({
            'Gender':             [mapeo_gender[gender]],
            'Married':            [mapeo_married[married]],
            'Dependents':         [mapeo_dep[dependents]],
            'Education':          [mapeo_edu[education]],
            'Self_Employed':      [mapeo_self[self_employed]],
            'ApplicantIncome':    [applicant_income],
            'CoapplicantIncome':  [coapplicant_income],
            'LoanAmount':         [loan_amount],
            'Loan_Amount_Term':   [loan_term],
            'Credit_History':     [credit_history],
            'Property_Area':      [mapeo_area[property_area]],
        })

        # ── Validaciones básicas ────────────────
        errores = []
        if applicant_income == 0 and coapplicant_income == 0:
            errores.append("⚠️ Al menos un solicitante debe tener ingresos.")
        if loan_amount > (applicant_income + coapplicant_income) * loan_term / 1000 * 2:
            errores.append("⚠️ El monto del préstamo parece muy alto en relación a los ingresos.")

        if errores:
            for e in errores:
                st.warning(e)

        # ── Predicción ──────────────────────────
        prediccion = modelo.predict(input_data)[0]
        probabilidades = modelo.predict_proba(input_data)[0]
        prob_aprobado  = probabilidades[1] * 100
        prob_rechazado = probabilidades[0] * 100

        # ── Mostrar resultado ───────────────────
        if prediccion == 1:
            st.markdown(f"""
            <div class='resultado-aprobado'>
                ✅ PRÉSTAMO APROBADO<br>
                <span style='font-size:0.65em; font-weight:normal'>
                    El modelo recomienda aprobar esta solicitud
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.success(f"Confianza del modelo: **{prob_aprobado:.1f}%**")
        else:
            st.markdown(f"""
            <div class='resultado-rechazado'>
                ❌ PRÉSTAMO RECHAZADO<br>
                <span style='font-size:0.65em; font-weight:normal'>
                    El modelo recomienda no aprobar esta solicitud
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"Confianza del modelo: **{prob_rechazado:.1f}%**")

        # ── Probabilidades ─────────────────────
        st.markdown("**📈 Probabilidades del modelo:**")
        st.progress(int(prob_aprobado), text=f"Aprobado: {prob_aprobado:.1f}%")
        st.progress(int(prob_rechazado), text=f"Rechazado: {prob_rechazado:.1f}%")

        # ── Factores clave ─────────────────────
        st.markdown("**🔑 Factores clave de tu solicitud:**")

        importancias = modelo.feature_importances_
        features     = list(input_data.columns)
        top_features = sorted(zip(features, importancias),
                              key=lambda x: x[1], reverse=True)[:4]

        for feat, imp in top_features:
            valor = input_data[feat].values[0]
            labels = {
                'Credit_History':    {1: "✅ Bueno", 0: "❌ Malo"},
                'Gender':            {1: "Male", 0: "Female"},
                'Married':           {1: "Sí", 0: "No"},
                'Education':         {0: "Universitario", 1: "No universitario"},
                'Self_Employed':     {1: "Sí", 0: "No"},
                'Property_Area':     {0: "Rural", 1: "Semiurbano", 2: "Urbano"},
            }
            display_val = labels.get(feat, {}).get(valor, f"{valor:,}")
            pct = imp * 100
            st.markdown(f"""
            <div class='info-card' style='padding:12px; margin:5px 0'>
                <span style='color:#aaa; font-size:0.85em'>{feat.replace('_', ' ')}</span>
                <span style='float:right; color:#fff; font-weight:bold'>{display_val}</span><br>
                <small style='color:#888'>Influencia: {pct:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)

        # ── Resumen de datos ingresados ─────────
        with st.expander("📄 Ver datos ingresados"):
            resumen = {
                "Género": gender,
                "Casado": married,
                "Dependientes": dependents,
                "Educación": education,
                "Autoempleado": self_employed,
                "Ingresos solicitante": f"${applicant_income:,}",
                "Ingresos co-sol.": f"${coapplicant_income:,}",
                "Monto préstamo": f"${loan_amount}k",
                "Plazo": f"{loan_term} meses",
                "Historial crediticio": "Bueno" if credit_history == 1 else "Malo",
                "Área propiedad": property_area,
            }
            for k, v in resumen.items():
                st.markdown(f"**{k}:** {v}")


# ─────────────────────────────────────────────
# Footer / Sidebar con info del modelo
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ Sobre el modelo")
    st.markdown("""
    **Algoritmo:** Árbol de Decisión  
    **Dataset:** Loan Approval Dataset  
    **Registros:** 614 reales  
    **Features:** 11 variables  
    """)

    st.markdown("---")
    st.markdown("**📚 ¿Qué es un Árbol de Decisión?**")
    st.info(
        "Es un modelo que aprende reglas de decisión a partir de los datos, "
        "similar a un diagrama de flujo. Cada nodo hace una pregunta y las "
        "ramas llevan a la respuesta final."
    )

    st.markdown("---")
    st.markdown("**🎓 Inteligencia Artificial I**")
    st.markdown("Actividad 3 · Despliegue en la nube")

    st.markdown("---")
    st.caption("Modelo entrenado con scikit-learn · Desplegado con Streamlit Cloud")
