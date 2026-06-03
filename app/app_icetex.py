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
    /* ── Base ── */
    .stApp {
        background-color: #F4F6F9;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ── Encabezado ── */
    .header-bar {
        background-color: #003C82;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .header-bar h1 {
        color: #ffffff !important;
        font-size: 1.6em;
        margin: 0;
        font-weight: 700;
    }
    .header-bar p {
        color: #A8C4E0 !important;
        margin: 4px 0 0 0;
        font-size: 0.9em;
    }

    /* ── Secciones del formulario ── */
    .section-card {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .section-title {
        font-size: 0.78em;
        font-weight: 700;
        color: #003C82;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }

    /* ── Tabla de rangos ── */
    .rangos-card {
        background: #EFF4FB;
        border: 1px solid #C5D8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .rangos-title {
        font-size: 0.78em;
        font-weight: 700;
        color: #003C82;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }

    /* ── Botón ── */
    .stButton > button {
        width: 100%;
        background-color: #003C82;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 14px 0;
        font-size: 1em;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #00529E;
        color: white;
    }

    /* ── Panel resultado ── */
    .result-panel {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 24px;
        height: 100%;
    }
    .result-title {
        font-size: 0.78em;
        font-weight: 700;
        color: #003C82;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }

    /* ── Tarjeta de resultado ── */
    .resultado-card {
        padding: 24px 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.25em;
        font-weight: 700;
        margin: 0 0 20px 0;
        color: #ffffff;
    }
    .res-matricula     { background-color: #0F6E56; border-left: 5px solid #085041; }
    .res-sostenimiento { background-color: #185FA5; border-left: 5px solid #0C447C; }
    .res-otro          { background-color: #534AB7; border-left: 5px solid #3C3489; }

    /* ── Barras de probabilidad ── */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .prob-label {
        font-size: 0.82em;
        color: #4A5568;
        width: 160px;
        flex-shrink: 0;
    }
    .prob-bar-bg {
        flex: 1;
        background: #E2E8F0;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        background-color: #003C82;
    }
    .prob-pct {
        font-size: 0.82em;
        font-weight: 600;
        color: #003C82;
        width: 36px;
        text-align: right;
        flex-shrink: 0;
    }

    /* ── Factor cards ── */
    .factor-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 12px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 7px;
        margin-bottom: 7px;
    }
    .factor-name {
        font-size: 0.8em;
        color: #718096;
    }
    .factor-val {
        font-size: 0.82em;
        font-weight: 600;
        color: #1A202C;
    }
    .factor-imp {
        font-size: 0.78em;
        color: #ffffff;
        background-color: #003C82;
        padding: 2px 8px;
        border-radius: 20px;
        margin-left: 8px;
    }

    /* ── Estado vacío ── */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #A0AEC0;
    }
    .empty-state .icon { font-size: 2.5em; margin-bottom: 12px; }
    .empty-state p { font-size: 0.88em; color: #A0AEC0 !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-badge {
        background: #EFF4FB;
        border: 1px solid #C5D8F0;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.83em;
        color: #2D3748;
    }
    .sidebar-badge strong { color: #003C82; }

    /* ── Overrides Streamlit ── */
    h1, h2, h3 { color: #1A202C !important; }
    label, .stSelectbox label, .stSlider label { color: #4A5568 !important; font-size: 0.88em !important; }
    p { color: #4A5568; }
    .stMarkdown hr { border-color: #E2E8F0; }
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
st.markdown("""
<div class="header-bar">
    <div style="font-size:2.2em;line-height:1">🎓</div>
    <div>
        <h1>Predictor de Modalidad de Crédito ICETEX</h1>
        <p>Árbol de Decisión &nbsp;·&nbsp; Datos reales Colombia &nbsp;·&nbsp; Inteligencia Artificial I</p>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("""
    ⚠️ **Modelo no encontrado.**
    1. Abre el notebook `notebooks/01_training_icetex.ipynb`
    2. Ejecuta todas las celdas (Cell → Run All)
    3. Verifica que exista `models/icetex_model.pkl`
    4. Recarga esta página
    """)
    st.stop()


# ─────────────────────────────────────────────
# Opciones desde encoders
# ─────────────────────────────────────────────
def opciones(col):
    return list(encoders[col].classes_)


# ─────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:

    # ── Datos personales ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">👤 Datos personales</div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sexo    = st.selectbox("Sexo al nacer", opciones('SEXO AL NACER'))
        estrato = st.select_slider("Estrato socioeconómico", options=[1,2,3,4,5,6], value=2)
    with c2:
        departamento  = st.selectbox("Departamento de origen", opciones('DEPARTAMENTO DE ORIGEN'))
        categoria_mun = st.selectbox("Categoría del municipio", opciones('CATEGORÍA DEL MUNICIPIO DE ORIGEN'))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Datos académicos ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📚 Datos académicos</div>
    """, unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        sector_ies      = st.selectbox("Sector de la IES", opciones('SECTOR IES'),
                                       help="Oficial = pública · Privada = privada")
        nivel_formacion = st.selectbox("Nivel de formación", opciones('NIVEL DE FORMACIÓN'))
    with c4:
        modalidad_linea = st.selectbox("Modalidad de línea", opciones('MODALIDAD DE LÍNEA'),
                                       help="Pregrado, Posgrado País o Crédito Exterior")
        rango_valor     = st.selectbox("Rango del valor desembolsado",
                                       opciones('RANGO DEL VALOR TOTAL DESEMBOLSADO'),
                                       help="Decil del valor total del crédito (I = menor · X = mayor)")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Tabla de rangos ──
    st.markdown('<div class="rangos-card"><div class="rangos-title">📊 Referencia — rangos de valor desembolsado</div>',
                unsafe_allow_html=True)
    st.caption("Deciles calculados por vigencia y modalidad de línea. Valores en COP aproximados.")
    st.dataframe(
        {
            "Decil": ["I","II","III","IV","V","VI","VII","VIII","IX","X"],
            "Rango aprox. (COP)": [
                "$0 — $1.500.000",
                "$1.500.001 — $3.000.000",
                "$3.000.001 — $5.000.000",
                "$5.000.001 — $8.000.000",
                "$8.000.001 — $12.000.000",
                "$12.000.001 — $17.000.000",
                "$17.000.001 — $23.000.000",
                "$23.000.001 — $32.000.000",
                "$32.000.001 — $50.000.000",
                "> $50.000.000",
            ],
            "Nivel típico": [
                "Técnico / SENA",
                "Técnico / Tecnólogo",
                "Universitario (IES oficial)",
                "Universitario",
                "Universitario (IES privada)",
                "Universitario (IES privada)",
                "Universitario / Especialización",
                "Posgrado país",
                "Maestría / Doctorado nacional",
                "Posgrado exterior",
            ],
        },
        use_container_width=True,
        hide_index=True,
        height=388,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    predecir = st.button("Predecir modalidad de crédito", use_container_width=True)


# ─────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────
with col_result:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">📊 Resultado de la predicción</div>', unsafe_allow_html=True)

    if not predecir:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🤖</div>
            <p><strong style="color:#718096">Esperando datos</strong></p>
            <p>Completa el formulario y presiona<br><em>Predecir modalidad de crédito</em></p>
        </div>
        """, unsafe_allow_html=True)

    else:
        def encode(col, valor):
            try:
                return int(encoders[col].transform([valor])[0])
            except Exception:
                return 0

        input_data = pd.DataFrame([{
            'SEXO AL NACER':                      encode('SEXO AL NACER', sexo),
            'ESTRATO SOCIOECONÓMICO':              estrato,
            'DEPARTAMENTO DE ORIGEN':             encode('DEPARTAMENTO DE ORIGEN', departamento),
            'CATEGORÍA DEL MUNICIPIO DE ORIGEN':  encode('CATEGORÍA DEL MUNICIPIO DE ORIGEN', categoria_mun),
            'SECTOR IES':                         encode('SECTOR IES', sector_ies),
            'NIVEL DE FORMACIÓN':                 encode('NIVEL DE FORMACIÓN', nivel_formacion),
            'MODALIDAD DE LÍNEA':                 encode('MODALIDAD DE LÍNEA', modalidad_linea),
            'RANGO DEL VALOR TOTAL DESEMBOLSADO': encode('RANGO DEL VALOR TOTAL DESEMBOLSADO', rango_valor),
        }])

        pred_num       = model.predict(input_data)[0]
        probabilidades = model.predict_proba(input_data)[0]
        modalidad_pred = encoders['MODALIDAD DEL CRÉDITO'].inverse_transform([pred_num])[0]

        color_class = 'res-matricula'     if 'MATRI' in modalidad_pred.upper() else \
                      'res-sostenimiento' if 'SOSTE' in modalidad_pred.upper() else \
                      'res-otro'

        # Tarjeta resultado
        st.markdown(f"""
        <div class="resultado-card {color_class}">
            {modalidad_pred}
            <div style="font-size:0.55em;font-weight:400;margin-top:6px;opacity:0.85">
                Modalidad predicha por el modelo
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilidades
        st.markdown("<p style='font-size:0.78em;font-weight:700;color:#003C82;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px'>Probabilidad por modalidad</p>",
                    unsafe_allow_html=True)

        probs_df = pd.DataFrame({
            'Modalidad':    encoders['MODALIDAD DEL CRÉDITO'].inverse_transform(range(len(probabilidades))),
            'Probabilidad': probabilidades
        }).sort_values('Probabilidad', ascending=False)

        for _, row in probs_df.iterrows():
            pct = int(row['Probabilidad'] * 100)
            es_pred = row['Modalidad'] == modalidad_pred
            weight = "700" if es_pred else "400"
            color  = "#003C82" if es_pred else "#718096"
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label" style="font-weight:{weight};color:{color}">
                    {row['Modalidad']}
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%;background-color:{color}"></div>
                </div>
                <div class="prob-pct" style="color:{color}">{pct}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Variables influyentes
        st.markdown("<p style='font-size:0.78em;font-weight:700;color:#003C82;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px'>Variables más influyentes</p>",
                    unsafe_allow_html=True)

        labels_legibles = {
            'SEXO AL NACER':                      sexo,
            'ESTRATO SOCIOECONÓMICO':              f'Estrato {estrato}',
            'DEPARTAMENTO DE ORIGEN':             departamento,
            'CATEGORÍA DEL MUNICIPIO DE ORIGEN':  categoria_mun,
            'SECTOR IES':                         sector_ies,
            'NIVEL DE FORMACIÓN':                 nivel_formacion,
            'MODALIDAD DE LÍNEA':                 modalidad_linea,
            'RANGO DEL VALOR TOTAL DESEMBOLSADO': f'Rango {rango_valor}',
        }

        feature_imp = sorted(
            zip(list(input_data.columns), model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:4]

        for feat, imp in feature_imp:
            st.markdown(f"""
            <div class="factor-row">
                <div>
                    <div class="factor-name">{feat}</div>
                    <div class="factor-val">{labels_legibles.get(feat, '')}</div>
                </div>
                <span class="factor-imp">{imp*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Ver todos los datos ingresados"):
            for k, v in labels_legibles.items():
                st.markdown(f"**{k}:** {v}")

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Sobre el modelo")

    st.markdown("""
    <div class="sidebar-badge">
        <strong>Algoritmo</strong><br>Árbol de Decisión
    </div>
    <div class="sidebar-badge">
        <strong>Fuente de datos</strong><br>ICETEX — Datos Abiertos Colombia
    </div>
    <div class="sidebar-badge">
        <strong>Registros de entrenamiento</strong><br>109.139 créditos reales
    </div>
    <div class="sidebar-badge">
        <strong>Variables de entrada</strong><br>8 features
    </div>
    <div class="sidebar-badge">
        <strong>Variable objetivo</strong><br>Modalidad del crédito
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info(
        "Dado el perfil de un estudiante colombiano, "
        "el modelo predice qué tipo de crédito ICETEX "
        "recibiría: Matrícula, Sostenimiento u otra modalidad."
    )

    st.markdown("---")
    st.caption("Inteligencia Artificial I · Actividad 3")
    st.caption("Modelo entrenado con scikit-learn · Streamlit")