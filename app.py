import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from PIL import Image
import os
import json
from fpdf import FPDF
from streamlit_image_zoom import image_zoom

# --- CONFIGURAÇÃO DE ARQUIVOS E ÍCONES ---
ISOTOPES_FILE = "isotopes.json"
ICON_FILE = "UERJ.ico"

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

# --- CONTROLE DE TEMA (CSS HACK) ---
def apply_theme_css(theme):
    if theme == "Escuro":
        st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                background-color: #0e1117;
                color: #fafafa;
            }
            [data-testid="stSidebar"] {
                background-color: #262730;
                color: #fafafa;
            }
            .stTextInput > div > div > input { color: black; }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff;
                color: #000000;
            }
            [data-testid="stSidebar"] {
                background-color: #f0f2f6;
                color: #000000;
            }
            </style>
            """, unsafe_allow_html=True)

# --- ESTILIZAÇÃO DE FONTE ---
st.markdown("""
    <style>
    .stMarkdown, .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stButton > button, .stTable, .stDataFrame {
        font-family: 'Times New Roman', Times, serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Georgia', 'Times New Roman', serif !important;
        font-weight: bold;
    }
    input, .stSelectbox div[data-baseweb="select"] span {
        font-family: 'Times New Roman', Times, serif !important;
    }
    code {
        font-family: 'Courier New', Courier, monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES ---
AVOGADRO_NUMBER = 6.02214076e23

DEFAULT_ISOTOPES = {
    "Césio-137": {"lambda": 0.02298, "half_life": 30.17, "unit": "anos", "atomic_weight": 136.907},
    "Bário-137m": {"lambda": 236.6, "half_life": 2.55, "unit": "minutos", "atomic_weight": 136.9},
    "Carbono-14": {"lambda": 1.20968e-4, "half_life": 5730, "unit": "anos", "atomic_weight": 14.003},
    "Tório-232": {"lambda": 4.95105e-11, "half_life": 1.4e10, "unit": "anos", "atomic_weight": 232.038},
    "Cobalto-60": {"lambda": 0.1315, "half_life": 5.27, "unit": "anos", "atomic_weight": 59.933},
    "Iodo-131": {"lambda": 31.55, "half_life": 8.02, "unit": "dias", "atomic_weight": 130.906},
    "Urânio-238": {"lambda": 1.551e-10, "half_life": 4.468e9, "unit": "anos", "atomic_weight": 238.050}
}

CONVERSIONS_TO_YEARS = {
    "segundos": 1 / (365.25 * 24 * 60 * 60),
    "minutos": 1 / (365.25 * 24 * 60),
    "horas": 1 / (365.25 * 24),
    "dias": 1 / 365.25,
    "anos": 1.0
}

# --- PERSISTÊNCIA ---
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

# --- CÁLCULOS MATEMÁTICOS ---
def convert_time_to_years(value, unit):
    return value * CONVERSIONS_TO_YEARS.get(unit, 1)

def calculate_simple_decay(N0, lam, t_years):
    return N0 * np.exp(-lam * t_years)

def mass_to_nuclei(mass_g, atomic_weight):
    if atomic_weight <= 0: return 0
    return (mass_g / atomic_weight) * AVOGADRO_NUMBER

def nuclei_to_mass(nuclei, atomic_weight):
    return (nuclei / AVOGADRO_NUMBER) * atomic_weight

# --- FUNÇÃO GERADORA DE PDF ---
def generate_pdf_report(df, title, t_unit):
    pdf = FPDF()
    pdf.add_page()
    
    # Título
    pdf.set_font("Times", "B", 16)
    pdf.cell(0, 10, f"Relatorio: {title}", ln=True, align="C")
    pdf.ln(10)
    
    # Configuração da Tabela
    pdf.set_font("Times", "B", 10)
    page_width = pdf.w - 2 * pdf.l_margin
    col_width = page_width / len(df.columns)
    row_height = 8

    # Cabeçalho
    for col in df.columns:
        clean_col = str(col).replace("(", "").replace(")", "").replace("e-", "E-")
        pdf.cell(col_width, row_height, clean_col[:20], border=1, align="C")
    pdf.ln()

    # Dados
    pdf.set_font("Times", "", 10)
    for index, row in df.iterrows():
        for item in row:
            if isinstance(item, (float, int)):
                txt = f"{item:.4e}"
            else:
                txt = str(item)
            pdf.cell(col_width, row_height, txt, border=1, align="C")
        pdf.ln()
    
    return pdf.output(dest="S").encode("latin-1", "replace")

# --- RENDERIZADOR: CALCULADORA ---
def render_calculator(chart_theme):
    st.title("Calculadora de Decaimento Radioativo")
    st.markdown("---")
    mode_tab1, mode_tab2 = st.tabs(["Decaimento Simples (A → Estável)", "Decaimento em Cadeia (Séries Naturais)"])

    with mode_tab1:
        run_simple_mode(chart_theme)

    with mode_tab2:
        run_chain_mode(chart_theme)

def run_simple_mode(chart_theme):
    col_config, col_results = st.columns([1, 2])
    
    with col_config:
        st.subheader("Parâmetros (Simples)")
        
        def update_lambda_callback():
            new_iso = st.session_state.simple_iso
            new_lambda = st.session_state.isotopes[new_iso]["lambda"]
            st.session_state.simple_lam = float(new_lambda)

        iso_list = list(st.session_state.isotopes.keys())
        idx_padrao = 0
        if "Césio-137" in iso_list:
            idx_padrao = iso_list.index("Césio-137")
            
        selected_iso = st.selectbox(
            "Isótopo", 
            iso_list, 
            index=idx_padrao, 
            key="simple_iso", 
            on_change=update_lambda_callback 
        )
        
        iso_data = st.session_state.isotopes[selected_iso]
        
        if "simple_lam" not in st.session_state:
            st.session_state.simple_lam = float(iso_data["lambda"])

        custom_lambda = st.number_input(
            "Lambda (anos⁻¹)", 
            format="%.4e", 
            key="simple_lam"
        )
        
        st.markdown("**Tempo**")
        c1, c2 = st.columns([2, 1])
        t_val = c1.number_input("Duração", value=100.0, key="simple_t")
        t_unit = c2.selectbox("Unidade", list(CONVERSIONS_TO_YEARS.keys()), index=4, key="simple_unit")
        
        st.markdown("**Quantidade Inicial**")
        input_mode = st.radio("Entrada:", ["Massa (g)", "Núcleos (N0)"], horizontal=True, key="simple_mode")
        
        N0 = 0
        atomic_w = iso_data["atomic_weight"]
        
        if input_mode == "Massa (g)":
            mass_initial = st.number_input("Massa (g)", value=1.0, format="%.4e", key="simple_mass")
            N0 = mass_to_nuclei(mass_initial, atomic_w)
        else:
            N0 = st.number_input("N0", value=1.0e20, format="%.4e", key="simple_n0")
            
        steps = st.slider("Passos (Intervalos)", 10, 500, 100, key="simple_steps")
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
        
        fig = go.Figure()
        color = '#00CC96' if chart_theme == 'plotly_dark' else '#1f77b4'
        hover_txt = [f"t={t:.2f}<br>Qtd={y:.4e} {unit_label}" for t, y in zip(t_plot, y_vals)]
        
        fig.add_trace(go.Scatter(
            x=t_plot, y=y_vals, mode='lines', name=selected_iso,
            line=dict(color=color, width=3),
            text=hover_txt, hoverinfo="text"
        ))
        
        setup_graph_layout(fig, f"Decaimento de {selected_iso}", t_unit, unit_label, log_scale, chart_theme, max_t)
        st.plotly_chart(fig, use_container_width=True)
        
        data_dict = {f"Quantidade ({unit_label})": y_vals}
        show_datatable(t_plot, data_dict, t_unit, "simple_dl", report_title=f"Decaimento {selected_iso}")

# --- NOVA FUNÇÃO: DECAIMENTO EM CADEIA (VISUAL) ---
# --- SUBSTITUA APENAS A FUNÇÃO run_chain_mode POR ESTA ---
def run_chain_mode(chart_theme):
    st.container()
    st.markdown("### Seleção de Séries Naturais")
    st.markdown("---")
    
    chain_images = {
        "Série do Urânio (U-238)": "uranium_chain.png",
        "Série do Tório (Th-232)": "thorium_chain.png",
        "Série do Actínio (U-235)": "actinium_chain.png"
    }

    # Mantendo o layout que joga a imagem para a direita
    col_select, col_space, col_image = st.columns([1, 0.3, 2])

    with col_select:
        st.subheader("Configuração")
        selected_chain = st.selectbox("Selecione a Cadeia:", list(chain_images.keys()))
        
        st.info(f"Cadeia Ativa:\n**{selected_chain}**")
        st.markdown("---")

    with col_image:
        st.subheader("Visualização da série")
        st.caption("🖱️ Use o scroll do mouse para zoom e clique para arrastar.")
        
        image_name = chain_images[selected_chain]
        
        img_path = None
        if os.path.exists(image_name):
            img_path = image_name
        elif os.path.exists(os.path.join("assets", image_name)):
            img_path = os.path.join("assets", image_name)
            
        if img_path:
            pil_image = Image.open(img_path)
            
            image_zoom(
                pil_image,
                mode="scroll",
                # AUMENTEI AQUI: De (700, 500) para (700, 700)
                size=(700, 700), 
                keep_aspect_ratio=True,
                zoom_factor=4.0,
                increment=0.2
            )
        else:
            st.error(f"Imagem '{image_name}' não encontrada.")

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
        xaxis=dict(
            range=[0, x_range_max],
            tickmode='array',
            tickvals=custom_ticks,
            ticktext=[f"{x:.1f}" for x in custom_ticks],
            constrain='domain'
        )
    )

def show_datatable(t_vec, data_cols, t_unit, key_prefix, report_title="Relatorio"):
    # Cria o DataFrame Base
    df_dict = {f"Tempo ({t_unit})": t_vec}
    df_dict.update(data_cols)
    df = pd.DataFrame(df_dict)
    
    # Formatação especial para CSV
    df_csv = df.copy()
    
    col_csv, col_pdf = st.columns(2)

    # 1. Download CSV
    csv = df_csv.to_csv(index=False).encode('utf-8')
    col_csv.download_button(
        label="📥 Baixar CSV",
        data=csv,
        file_name="dados_decaimento.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv"
    )

    # 2. Download PDF
    try:
        pdf_bytes = generate_pdf_report(df, report_title, t_unit)
        col_pdf.download_button(
            label="📄 Baixar PDF",
            data=pdf_bytes,
            file_name="relatorio_decaimento.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf"
        )
    except Exception as e:
        col_pdf.error(f"Erro ao gerar PDF: {e}")
    
    st.dataframe(
        df,
        use_container_width=True,
        height=300,
        hide_index=True
    )

def render_manager():
    st.title("Gerenciador de Isótopos")
    
    df_iso = pd.DataFrame.from_dict(st.session_state.isotopes, orient='index')
    df_iso.reset_index(inplace=True)
    df_iso.rename(columns={'index': 'Nome', 'lambda': 'Lambda', 'half_life': 'Meia-vida', 'atomic_weight': 'Massa Atômica'}, inplace=True)
    
    st.dataframe(
        df_iso,
        column_config={
            'Lambda': st.column_config.NumberColumn(format="%.4e"),
            'Meia-vida': st.column_config.NumberColumn(format="%.4e"),
            'Massa Atômica': st.column_config.NumberColumn(format="%.4f")
        },
        use_container_width=True, 
        hide_index=True
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Adicionar")
        with st.form("add"):
            name = st.text_input("Nome")
            mass = st.number_input("Massa", 1.0, format="%.4f")
            hl = st.number_input("Meia-vida", 1.0)
            unit = st.selectbox("Unidade", list(CONVERSIONS_TO_YEARS.keys()), index=4)
            if st.form_submit_button("Salvar"):
                if name:
                    hl_years = convert_time_to_years(hl, unit)
                    lam = np.log(2)/hl_years if hl_years > 0 else 0
                    st.session_state.isotopes[name] = {
                        "lambda": lam, "half_life": hl, "unit": unit, "atomic_weight": mass
                    }
                    save_isotopes_to_file(st.session_state.isotopes)
                    st.success("Adicionado!")
                    st.rerun()

    with c2:
        st.subheader("Remover")
        to_del = st.selectbox("Isótopo", list(st.session_state.isotopes.keys()))
        if st.button("Remover"):
            if len(st.session_state.isotopes) > 1:
                del st.session_state.isotopes[to_del]
                save_isotopes_to_file(st.session_state.isotopes)
                st.success("Removido!")
                st.rerun()
            else:
                st.error("Mínimo 1 isótopo necessário.")
        
        if st.button("Restaurar Padrões"):
            st.session_state.isotopes = DEFAULT_ISOTOPES.copy()
            save_isotopes_to_file(st.session_state.isotopes)
            st.rerun()

with st.sidebar:
    c1, logo, c2 = st.columns([1, 2, 1])
    with logo:
        if app_icon: st.image(app_icon, width=120)
        
    st.markdown("<h3 style='text-align: center; font-family: Georgia;'>UERJ - Ciência da Computação</h3>", unsafe_allow_html=True)
    page = st.radio("Menu", ["Calculadora", "Gerenciar Isótopos"])
    st.markdown("---")
    
    # SELETOR DE TEMA
    theme = st.radio("Tema Gráfico", ["Escuro", "Claro"], horizontal=True)
    
    # Aplica o CSS para a interface inteira
    apply_theme_css(theme)
    
    # Define o tema do gráfico Plotly
    chart_theme = "plotly_dark" if theme == "Escuro" else "plotly"
    
    st.caption(f"© {date.today().year} UERJ")

if page == "Calculadora": render_calculator(chart_theme)
elif page == "Gerenciar Isótopos": render_manager()