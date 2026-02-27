import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
from PIL import Image
import os
import json
from fpdf import FPDF

# --- TENTATIVA DE IMPORTAR PLOTLY ---
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- CONFIGURAÇÃO DE ARQUIVOS ---
ISOTOPES_FILE = "isotopes.json"
ICON_FILE = "UERJ.ico"
CHAIN_IMAGE_FILE = "uranium_chain.png" 

if not os.path.exists(ICON_FILE):
    ICON_FILE = os.path.join("assets", "UERJ.ico")
if os.path.exists(ICON_FILE):
    app_icon = Image.open(ICON_FILE)
else:
    app_icon = None

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Calculadora de Decaimento UERJ",
    page_icon=app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS / TEMA ---
def apply_theme_css(theme):
    if theme == "Escuro":
        st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #fafafa; }
            [data-testid="stSidebar"] { background-color: #262730; color: #fafafa; }
            .stTextInput > div > div > input { color: black; }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #000000; }
            [data-testid="stSidebar"] { background-color: #f0f2f6; color: #000000; }
            </style>
            """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .stMarkdown, .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stButton > button, .stTable, .stDataFrame {
        font-family: 'Times New Roman', Times, serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Georgia', 'Times New Roman', serif !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES ---
AVOGADRO_NUMBER = 6.02214076e23

CONVERSIONS_TO_YEARS = {
    "segundos": 1 / (365.25 * 24 * 60 * 60),
    "minutos": 1 / (365.25 * 24 * 60),
    "horas": 1 / (365.25 * 24),
    "dias": 1 / 365.25,
    "anos": 1.0
}

# --- BANCO DE DADOS COMPLETO (Unidades: ANOS) ---
DEFAULT_ISOTOPES = {
    "Césio-137":  {"lambda": 0.02298,   "half_life": 30.17,    "unit": "anos", "atomic_weight": 136.907},
    "Bário-137m": {"lambda": 142916.0,  "half_life": 4.85e-6,  "unit": "anos", "atomic_weight": 136.9},
    "Carbono-14": {"lambda": 1.209e-4,  "half_life": 5730.0,   "unit": "anos", "atomic_weight": 14.003},
    "Cobalto-60": {"lambda": 0.1315,    "half_life": 5.27,     "unit": "anos", "atomic_weight": 59.933},
    "Iodo-131":   {"lambda": 31.56,     "half_life": 0.02195,  "unit": "anos", "atomic_weight": 130.906},
    "U-238":      {"lambda": 1.5403e-10, "half_life": 4.5e9,     "unit": "anos", "atomic_weight": 238.05},
    "Th-234":     {"lambda": 10.504,     "half_life": 0.06598,   "unit": "anos", "atomic_weight": 234.04},
    "Pa-234":     {"lambda": 311544.0,   "half_life": 2.22e-6,   "unit": "anos", "atomic_weight": 234.04},
    "U-234":      {"lambda": 2.8234e-06, "half_life": 245500.0,  "unit": "anos", "atomic_weight": 234.04},
    "Th-230":     {"lambda": 9.1954e-06, "half_life": 75380.0,   "unit": "anos", "atomic_weight": 230.03},
    "Ra-226":     {"lambda": 4.3267e-04, "half_life": 1602.0,    "unit": "anos", "atomic_weight": 226.02},
    "Rn-222":     {"lambda": 66.626,     "half_life": 0.0104,    "unit": "anos", "atomic_weight": 222.01},
    "Po-218":     {"lambda": 117548.0,   "half_life": 5.89e-6,   "unit": "anos", "atomic_weight": 218.00},
    "Pb-214":     {"lambda": 13598.0,    "half_life": 5.09e-5,   "unit": "anos", "atomic_weight": 214.00},
    "Bi-214":     {"lambda": 18221.0,    "half_life": 3.80e-5,   "unit": "anos", "atomic_weight": 213.99},
    "Ti-210":     {"lambda": 1.334e+11,  "half_life": 5.19e-12,  "unit": "anos", "atomic_weight": 213.99},
    "Pb-210":     {"lambda": 0.03108,    "half_life": 22.3,      "unit": "anos", "atomic_weight": 209.98},
    "Bi-210":     {"lambda": 50.636,     "half_life": 0.0137,    "unit": "anos", "atomic_weight": 209.98},
    "Po-210":     {"lambda": 1.8336,     "half_life": 0.3778,    "unit": "anos", "atomic_weight": 209.98},
    "Pb-206":     {"lambda": 0.0,        "half_life": 0.0,       "unit": "anos", "atomic_weight": 205.97}
}

# --- DADOS VISUAIS (Rótulos Limpos) ---
URANIUM_SERIES_DATA = {
    "U-238":  {"half_life_label": "4.5 Bilhões Anos", "lambda": 1.5403e-10},
    "Th-234": {"half_life_label": "0.066 Anos",       "lambda": 10.504},
    "Pa-234": {"half_life_label": "2.22e-6 Anos",     "lambda": 311544.0},
    "U-234":  {"half_life_label": "245500 Anos",      "lambda": 2.8234e-06},
    "Th-230": {"half_life_label": "75380 Anos",       "lambda": 9.1954e-06},
    "Ra-226": {"half_life_label": "1602 Anos",        "lambda": 4.3267e-04},
    "Rn-222": {"half_life_label": "0.0104 Anos",      "lambda": 66.626},
    "Po-218": {"half_life_label": "5.9e-6 Anos",      "lambda": 117548.0},
    "Pb-214": {"half_life_label": "5.1e-5 Anos",      "lambda": 13598.0},
    "Bi-214": {"half_life_label": "3.8e-5 Anos",      "lambda": 18221.0},
    "Tl-210": {"half_life_label": "2.5e-6 Anos",      "lambda": 280329.0},
    "Pb-210": {"half_life_label": "22.3 Anos",        "lambda": 0.03108},
    "Bi-210": {"half_life_label": "0.0137 Anos",      "lambda": 50.636},
    "Po-210": {"half_life_label": "0.378 Anos",       "lambda": 1.8336},
    "Pb-206": {"half_life_label": "Estável",          "lambda": 0.0}
}

URANIUM_SERIES_ORDER = [
    "U-238", "Th-234", "Pa-234", "U-234", "Th-230", "Ra-226", 
    "Rn-222", "Po-218", "Pb-214", "Bi-214", "Tl-210", "Pb-210", 
    "Bi-210", "Po-210", "Pb-206"
]

# --- FUNÇÕES ---
def load_isotopes_from_file():
    if os.path.exists(ISOTOPES_FILE):
        try:
            with open(ISOTOPES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_ISOTOPES.copy()

def save_isotopes_to_file(data):
    try:
        with open(ISOTOPES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")

if "isotopes" not in st.session_state:
    st.session_state.isotopes = load_isotopes_from_file()

def convert_time_to_years(value, unit):
    return value * CONVERSIONS_TO_YEARS.get(unit, 1)

def calculate_simple_decay(N0, lam, t_years):
    return N0 * np.exp(-lam * t_years)

def mass_to_nuclei(mass_g, atomic_weight):
    if atomic_weight <= 0: return 0
    return (mass_g / atomic_weight) * AVOGADRO_NUMBER

def nuclei_to_mass(nuclei, atomic_weight):
    return (nuclei / AVOGADRO_NUMBER) * atomic_weight

def generate_pdf_report(df, title, t_unit):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", "B", 16)
    pdf.cell(0, 10, f"Relatorio: {title}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Times", "B", 10)
    page_width = pdf.w - 2 * pdf.l_margin
    col_width = page_width / max(len(df.columns), 1)
    row_height = 8
    for col in df.columns:
        clean_col = str(col).replace("(", "").replace(")", "").replace("e-", "E-")
        pdf.cell(col_width, row_height, clean_col[:15], border=1, align="C")
    pdf.ln()
    pdf.set_font("Times", "", 10)
    for index, row in df.iterrows():
        for item in row:
            txt = f"{item:.4e}" if isinstance(item, (float, int)) else str(item)
            pdf.cell(col_width, row_height, txt, border=1, align="C")
        pdf.ln()
    return pdf.output(dest="S").encode("latin-1", "replace")

def setup_graph_layout(fig, title, x_unit, y_unit, is_log, theme, max_x):
    custom_ticks = np.linspace(0, max_x, 6)
    x_range_max = max_x * 1.05
    fig.update_layout(
        title=dict(text=title, font=dict(family="Georgia", size=20)),
        xaxis_title=f"Tempo ({x_unit})",
        yaxis_title=f"Quantidade ({y_unit})",
        yaxis_type="log" if is_log else "linear",
        template=theme,
        height=500,
        hovermode="x unified",
        font=dict(family="Times New Roman", size=14),
        yaxis=dict(autorange=True),
        xaxis=dict(range=[0, x_range_max], tickmode='array', tickvals=custom_ticks, ticktext=[f"{x:.1f}" for x in custom_ticks], constrain='domain')
    )

# --- INTERFACE ---
def render_calculator(chart_theme):
    st.title("Calculadora de Decaimento Radioativo")
    st.markdown("---")
    
    mode_tab1, mode_tab2 = st.tabs(["Decaimento Simples (A → Estável)", "Decaimento em Cadeia (Visualização)"])

    with mode_tab1:
        run_simple_mode(chart_theme)

    with mode_tab2:
        run_chain_mode_visual()

def run_simple_mode(chart_theme):
    col_config, col_results = st.columns([1, 2])
    
    with col_config:
        st.subheader("Parâmetros (Simples)")
        
        def update_lambda_callback():
            new_iso = st.session_state.simple_iso
            new_lambda = st.session_state.isotopes[new_iso]["lambda"]
            st.session_state.simple_lam = float(new_lambda)
            
            # Salva o isótopo de forma segura
            st.session_state.iso_escolhido = new_iso

        iso_list = list(st.session_state.isotopes.keys())

        # 1. Se a variável protegida não existir, cria com padrão
        if "iso_escolhido" not in st.session_state:
            st.session_state.iso_escolhido = "Césio-137" if "Césio-137" in iso_list else iso_list[0]

        # 2. Descobre o índice correto do isótopo salvo
        idx_padrao = 0
        if st.session_state.iso_escolhido in iso_list:
            idx_padrao = iso_list.index(st.session_state.iso_escolhido)
            
        # 3. Monta o selectbox (único!) usando o índice certo
        selected_iso = st.selectbox("Isótopo", iso_list, index=idx_padrao, key="simple_iso", on_change=update_lambda_callback)
        
        # Pega os dados do isótopo selecionado
        iso_data = st.session_state.isotopes[selected_iso]
        
        if "simple_lam" not in st.session_state:
            st.session_state.simple_lam = float(iso_data["lambda"])

        custom_lambda = st.number_input("Lambda (anos⁻¹)", format="%.4e", key="simple_lam")
        
        saved_hl = iso_data.get('half_life', 0)
        saved_unit = iso_data.get('unit', 'anos')
        st.caption(f"Registro Salvo: Meia-vida = {saved_hl:.4e} {saved_unit}")

        st.markdown("---")
        st.markdown("**Tempo de Simulação**")
        
        c1, c2 = st.columns([2, 1])
        t_val = c1.number_input("Duração", value=100.0, key="simple_t", format="%.2f")
        t_unit = c2.selectbox("Unidade", list(CONVERSIONS_TO_YEARS.keys()), index=4, key="simple_unit")
        
        st.markdown("**Qtd Inicial**")
        input_mode = st.radio("Entrada:", ["Massa (g)", "Núcleos (N0)"], horizontal=True, key="simple_mode")
        
        N0 = 0
        atomic_w = iso_data["atomic_weight"]
        
        if input_mode == "Massa (g)":
            mass_initial = st.number_input("Massa (g)", value=1.0, format="%.4e", key="simple_mass")
            N0 = mass_to_nuclei(mass_initial, atomic_w)
        else:
            N0 = st.number_input("N0", value=1.0e20, format="%.4e", key="simple_n0")
            
        steps = st.slider("Passos", 10, 500, 100, key="simple_steps")
        log_scale = st.checkbox("Escala Log (Y)", value=False, key="simple_log")

    t_years_total = convert_time_to_years(t_val, t_unit)
    Nt_final = calculate_simple_decay(N0, custom_lambda, t_years_total)
    
    max_t = t_val if t_val > 0 else 100
    t_plot = np.linspace(0, max_t, steps + 1)
    t_years_vec = [convert_time_to_years(x, t_unit) for x in t_plot]
    Nt_vec = calculate_simple_decay(N0, custom_lambda, np.array(t_years_vec))
    
    y_vals = Nt_vec
    res_display = Nt_final
    unit_label = "Núcleos"
    if input_mode == "Massa (g)":
        y_vals = nuclei_to_mass(Nt_vec, atomic_w)
        res_display = nuclei_to_mass(Nt_final, atomic_w)
        unit_label = "g"

    with col_results:
        st.markdown(f"#### Resultado Final: {res_display:.4e} {unit_label}")
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            color = '#00CC96' if chart_theme == 'plotly_dark' else '#1f77b4'
            hover_txt = [f"t={t:.2f}<br>Qtd={y:.4e} {unit_label}" for t, y in zip(t_plot, y_vals)]
            fig.add_trace(go.Scatter(x=t_plot, y=y_vals, mode='lines', name=selected_iso, line=dict(color=color, width=3), text=hover_txt, hoverinfo="text"))
            setup_graph_layout(fig, f"Decaimento de {selected_iso}", t_unit, unit_label, log_scale, chart_theme, max_t)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Biblioteca 'plotly' não encontrada.")

        st.subheader("Tabela de Dados")
        df = pd.DataFrame({f"Tempo ({t_unit})": t_plot, f"Quantidade ({unit_label})": y_vals})
        
        c_csv, c_pdf = st.columns(2)
        c_csv.download_button("📥 CSV", df.to_csv(index=False).encode('utf-8'), "dados.csv", "text/csv")
        try:
            c_pdf.download_button("📄 PDF", generate_pdf_report(df, f"Decaimento {selected_iso}", t_unit), "relatorio.pdf", "application/pdf")
        except Exception as e:
            st.error(f"Erro PDF: {e}")
            
        # FORMATO "%.4e": NOTAÇÃO CIENTÍFICA FORÇADA
        st.dataframe(
            df,
            use_container_width=True,
            height=300,
            hide_index=True,
            column_config={
                f"Tempo ({t_unit})": st.column_config.NumberColumn(format="%.4e"),
                f"Quantidade ({unit_label})": st.column_config.NumberColumn(format="%.4e")
            }
        )

def run_chain_mode_visual():
    st.markdown("### Série de Decaimento do Urânio-238")
    col_config, col_image = st.columns([1, 1.5])
    
    with col_config:
        st.subheader("Configuração da Seleção")
        start_element = st.selectbox("Começar em (Pai):", URANIUM_SERIES_ORDER[:-1], index=0)
        start_idx = URANIUM_SERIES_ORDER.index(start_element)
        available_ends = URANIUM_SERIES_ORDER[start_idx+1:]
        end_element = st.selectbox("Terminar em (Filho):", available_ends, index=len(available_ends)-1)
        end_idx = URANIUM_SERIES_ORDER.index(end_element)
        
        selected_chain = URANIUM_SERIES_ORDER[start_idx : end_idx+1]
        st.info(f"Trecho selecionado: **{' → '.join(selected_chain)}**")
        st.markdown("---")
        
        hl_pai = URANIUM_SERIES_DATA[start_element]["half_life_label"]
        st.write(f"Meia-vida de **{start_element}**: {hl_pai}")
        st.caption("Esta aba é apenas para visualização da cadeia.")

    with col_image:
        st.subheader("Mapa da Série Natural")
        img_path = CHAIN_IMAGE_FILE
        if not os.path.exists(img_path): img_path = os.path.join("assets", CHAIN_IMAGE_FILE)
        if os.path.exists(img_path):
            st.image(img_path, caption="Série de decaimento U-238", use_container_width=True)
        else:
            st.warning(f"⚠️ Imagem não encontrada: `{CHAIN_IMAGE_FILE}`")

def render_manager():
    st.title("Gerenciador de Isótopos")
    st.info("💡 Clique em 'Restaurar Padrões' para carregar a tabela completa se ela estiver incompleta.")
    
    df_iso = pd.DataFrame.from_dict(st.session_state.isotopes, orient='index')
    df_iso.reset_index(inplace=True)
    df_iso.rename(columns={'index': 'Nome', 'lambda': 'Lambda (ano⁻¹)', 'half_life': 'Meia-vida', 'unit': 'Unidade'}, inplace=True)
    
    # FORMATO "%.4e": NOTAÇÃO CIENTÍFICA FORÇADA NO GERENCIADOR TAMBÉM
    st.dataframe(
        df_iso, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Lambda (ano⁻¹)": st.column_config.NumberColumn(format="%.4e"),
            "Meia-vida": st.column_config.NumberColumn(format="%.4e"),
            "Massa Atômica": st.column_config.NumberColumn(format="%.4f")
        }
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Adicionar (Use Ponto .)")
        with st.form("add"):
            name = st.text_input("Nome")
            mass = st.number_input("Massa Atômica", 1.0, format="%.4f")
            hl = st.number_input("Meia-vida (em ANOS)", 1.0, format="%.4e")
            st.caption("Unidade fixada em: Anos")
            if st.form_submit_button("Salvar"):
                if name:
                    lam = np.log(2)/hl if hl > 0 else 0
                    st.session_state.isotopes[name] = {"lambda": lam, "half_life": hl, "unit": "anos", "atomic_weight": mass}
                    save_isotopes_to_file(st.session_state.isotopes)
                    st.success("Salvo!")
                    st.rerun()
    with c2:
        st.subheader("Restaurar Banco de Dados")
        st.write("Caso a lista de isótopos esteja incompleta ou você queira voltar aos valores originais, clique no botão abaixo.")
        
        if st.button("Restaurar Padrões (Recomendado)"):
            st.session_state.isotopes = DEFAULT_ISOTOPES.copy()
            save_isotopes_to_file(st.session_state.isotopes)
            st.success("Banco de dados restaurado com sucesso!")
            st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    c1, logo, c2 = st.columns([1, 2, 1])
    with logo:
        if app_icon: st.image(app_icon, width=120)
    st.markdown("<h3 style='text-align: center; font-family: Georgia;'>UERJ - Ciência da Computação</h3>", unsafe_allow_html=True)
    page = st.radio("Menu", ["Calculadora", "Gerenciar Isótopos"])
    st.markdown("---")
    theme = st.radio("Tema", ["Escuro", "Claro"], horizontal=True)
    apply_theme_css(theme)
    chart_theme = "plotly_dark" if theme == "Escuro" else "plotly"
    st.caption(f"© {date.today().year} UERJ")

if page == "Calculadora": render_calculator(chart_theme)
elif page == "Gerenciar Isótopos": render_manager()