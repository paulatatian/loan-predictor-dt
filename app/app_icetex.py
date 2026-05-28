"""
Aplicación web — Predictor de Modalidad de Crédito ICETEX
Algoritmo: Árbol de Decisión
Datos: Colombia — ICETEX Créditos Otorgados
Inteligencia Artificial I - Actividad 3
"""

import streamlit as st
import joblib
import pandas as pd
import os

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor Crédito ICETEX",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    .resultado-card {
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 1.5em;
        font-weight: bold;
        margin: 16px 0;
    }
    .res-matricula   { background: linear-gradient(135deg, #11998e, #38ef7d); box-shadow: 0 8px 24px rgba(56,239,125,0.3); }
    .res-sostenimiento { background: linear-gradient(135deg, #2193b0, #6dd5ed); box-shadow: 0 8px 24px rgba(109,213,237,0.3); }
    .res-otro        { background: linear-gradient(135deg, #834d9b, #d04ed6); box-shadow: 0 8px 24px rgba(208,78,214,0.3); }

    .info-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 14px;
        margin: 6px 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 0;
        font-size: 1.1em;
        font-weight: bold;
    }
    h1, h2, h3 { color: white !important; }
    label { color: #ddd !important; }
    p { color: #ccc; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cargar modelo y encoders
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    try:
        model    = joblib.load(os.path.join(base, 'icetex_model.pkl'))
        encoders = joblib.load(os.path.join(base, 'label_encoders.pkl'))
        clases   = joblib.load(os.path.join(base, 'clases.pkl'))
        return model, encoders, clases
    except FileNotFoundError:
        return None, None, None

model, encoders, clases = cargar_modelo()


# ─────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    st.markdown("<div style='font-size:4em;text-align:center;padding-top:10px'>🎓</div>",
                unsafe_allow_html=True)
with col_titulo:
    st.markdown("<h1 style='margin-bottom:0'>Predictor de Modalidad de Crédito ICETEX</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa'>Árbol de Decisión · Datos reales Colombia · Inteligencia Artificial I</p>",
                unsafe_allow_html=True)

st.markdown("---")

if model is None:
    st.error("""
    ⚠️ **Modelo no encontrado**
    1. Abre el notebook `notebooks/01_training_icetex.ipynb`
    2. Ejecuta todas las celdas (Cell → Run All)
    3. Verifica que exista `models/icetex_model.pkl`
    4. Recarga esta página
    """)
    st.stop()


# ─────────────────────────────────────────────
# Obtener opciones reales de los encoders
# ─────────────────────────────────────────────
def opciones(col):
    return list(encoders[col].classes_)


# ─────────────────────────────────────────────
# Layout: formulario | resultado
# ─────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:
    st.markdown("### 📋 Datos del Solicitante")
    st.markdown("<p>Completa la información para predecir la modalidad del crédito.</p>",
                unsafe_allow_html=True)

    # Datos personales
    st.markdown("**👤 Datos personales**")
    c1, c2 = st.columns(2)
    with c1:
        sexo = st.selectbox("Sexo al nacer", opciones('SEXO AL NACER'))
        estrato = st.select_slider(
            "Estrato socioeconómico",
            options=[1, 2, 3, 4, 5, 6],
            value=2
        )
    with c2:
        departamento = st.selectbox("Departamento de origen",
                                    opciones('DEPARTAMENTO DE ORIGEN'))
        categoria_mun = st.selectbox("Categoría del municipio",
                                     opciones('CATEGORÍA DEL MUNICIPIO DE ORIGEN'))

    st.markdown("---")

    # Datos académicos
    st.markdown("**📚 Datos académicos**")
    c3, c4 = st.columns(2)
    with c3:
        sector_ies = st.selectbox("Sector de la IES",
                                  opciones('SECTOR IES'),
                                  help="Oficial = pública, Privada = privada")
        nivel_formacion = st.selectbox("Nivel de formación",
                                       opciones('NIVEL DE FORMACIÓN'))
    with c4:
        modalidad_linea = st.selectbox("Modalidad de línea",
                                       opciones('MODALIDAD DE LÍNEA'),
                                       help="Pregrado, Posgrado País o Crédito Exterior")
        rango_valor = st.selectbox("Rango del valor desembolsado",
                                   opciones('RANGO DEL VALOR TOTAL DESEMBOLSADO'),
                                   help="Decil del valor total del crédito (I=menor, X=mayor)")

    st.markdown("")
    predecir = st.button("🔍 Predecir Modalidad de Crédito", use_container_width=True)


# ─────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Resultado de la Predicción")

    if not predecir:
        st.markdown("""
        <div class='info-card' style='text-align:center;padding:50px 20px'>
            <div style='font-size:3em'>🤖</div>
            <h3 style='color:#aaa !important'>Esperando datos...</h3>
            <p style='color:#888'>Completa el formulario y presiona<br><b>Predecir Modalidad</b></p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Codificar inputs usando los mismos encoders del entrenamiento
        def encode(col, valor):
            try:
                return int(encoders[col].transform([valor])[0])
            except Exception:
                return 0

        input_data = pd.DataFrame([{
            'SEXO AL NACER':                     encode('SEXO AL NACER', sexo),
            'ESTRATO SOCIOECONÓMICO':             estrato,
            'DEPARTAMENTO DE ORIGEN':            encode('DEPARTAMENTO DE ORIGEN', departamento),
            'CATEGORÍA DEL MUNICIPIO DE ORIGEN': encode('CATEGORÍA DEL MUNICIPIO DE ORIGEN', categoria_mun),
            'SECTOR IES':                        encode('SECTOR IES', sector_ies),
            'NIVEL DE FORMACIÓN':                encode('NIVEL DE FORMACIÓN', nivel_formacion),
            'MODALIDAD DE LÍNEA':                encode('MODALIDAD DE LÍNEA', modalidad_linea),
            'RANGO DEL VALOR TOTAL DESEMBOLSADO': encode('RANGO DEL VALOR TOTAL DESEMBOLSADO', rango_valor),
        }])

        # Predicción
        pred_num       = model.predict(input_data)[0]
        probabilidades = model.predict_proba(input_data)[0]
        modalidad_pred = encoders['MODALIDAD DEL CRÉDITO'].inverse_transform([pred_num])[0]

        # Color según modalidad
        color_class = 'res-matricula' if 'MATRI' in modalidad_pred.upper() else \
                      'res-sostenimiento' if 'SOSTE' in modalidad_pred.upper() else \
                      'res-otro'

        st.markdown(f"""
        <div class='resultado-card {color_class}'>
            🎓 {modalidad_pred}<br>
            <span style='font-size:0.6em;font-weight:normal'>
                Modalidad de crédito predicha por el modelo
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Probabilidades por clase
        st.markdown("**📈 Probabilidad por modalidad:**")
        probs_df = pd.DataFrame({
            'Modalidad': encoders['MODALIDAD DEL CRÉDITO'].inverse_transform(
                range(len(probabilidades))),
            'Probabilidad': probabilidades
        }).sort_values('Probabilidad', ascending=False)

        for _, row in probs_df.iterrows():
            pct = int(row['Probabilidad'] * 100)
            es_pred = row['Modalidad'] == modalidad_pred
            label = f"**{row['Modalidad']}**" if es_pred else row['Modalidad']
            st.progress(pct, text=f"{label}: {pct}%")

        # Factores más influyentes
        st.markdown("**🔑 Variables más influyentes en este modelo:**")
        feature_imp = sorted(
            zip(list(input_data.columns), model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:4]

        labels_legibles = {
            'SEXO AL NACER': sexo,
            'ESTRATO SOCIOECONÓMICO': f'Estrato {estrato}',
            'DEPARTAMENTO DE ORIGEN': departamento,
            'CATEGORÍA DEL MUNICIPIO DE ORIGEN': categoria_mun,
            'SECTOR IES': sector_ies,
            'NIVEL DE FORMACIÓN': nivel_formacion,
            'MODALIDAD DE LÍNEA': modalidad_linea,
            'RANGO DEL VALOR TOTAL DESEMBOLSADO': f'Rango {rango_valor}',
        }

        for feat, imp in feature_imp:
            st.markdown(f"""
            <div class='info-card' style='padding:10px;margin:4px 0'>
                <span style='color:#aaa;font-size:0.85em'>{feat}</span>
                <span style='float:right;color:#fff;font-weight:bold'>{labels_legibles.get(feat,'')}</span><br>
                <small style='color:#888'>Influencia: {imp*100:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)

        # Resumen
        with st.expander("📄 Ver datos ingresados"):
            for k, v in labels_legibles.items():
                st.markdown(f"**{k}:** {v}")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ Sobre el modelo")
    st.markdown("""
    **Algoritmo:** Árbol de Decisión
    **Fuente de datos:** ICETEX Colombia
    **Registros:** 109.139 reales
    **Features:** 8 variables
    **Variable objetivo:** Modalidad del crédito
    """)

    st.markdown("---")
    st.markdown("**🇨🇴 ¿Qué predice?**")
    st.info(
        "Dado el perfil de un estudiante colombiano, "
        "el modelo predice qué tipo de crédito ICETEX "
        "recibiría: Matrícula, Sostenimiento u otra modalidad."
    )

    st.markdown("---")
    st.markdown("**🎓 Inteligencia Artificial I**")
    st.markdown("Actividad 3 · Datos reales Colombia")
    st.caption("Modelo entrenado con scikit-learn · Streamlit Cloud")
