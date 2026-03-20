import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
from PIL import Image
import os
import json
from fpdf import FPDF
from numpy.linalg import eig, inv

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

# --- BANCO DE DADOS COMPLETO ---
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
    "Tl-210":     {"lambda": 280329.0,   "half_life": 2.5e-6,    "unit": "anos", "atomic_weight": 213.99},
    "Pb-210":     {"lambda": 0.03108,    "half_life": 22.3,      "unit": "anos", "atomic_weight": 209.98},
    "Bi-210":     {"lambda": 50.636,     "half_life": 0.0137,    "unit": "anos", "atomic_weight": 209.98},
    "Po-210":     {"lambda": 1.8336,     "half_life": 0.3778,    "unit": "anos", "atomic_weight": 209.98},
    "Pb-206":     {"lambda": 0.0,        "half_life": 0.0,       "unit": "anos", "atomic_weight": 205.97}
}

URANIUM_SERIES_DATA = DEFAULT_ISOTOPES.copy()

URANIUM_SERIES_ORDER = [
    "U-238", "Th-234", "Pa-234", "U-234", "Th-230", "Ra-226", 
    "Rn-222", "Po-218", "Pb-214", "Bi-214", "Tl-210", "Pb-210", 
    "Bi-210", "Po-210", "Pb-206"
]

# --- FUNÇÕES NÚCLEO (MATRIZES E CACHE) ---
@st.cache_data
def precompute_decay_matrices(lambdas):
    n = len(lambdas)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -lambdas[i]
        if i > 0:
            A[i, i-1] = lambdas[i-1]
    
    D, X = eig(A)
    X_inv = inv(X)
    return D, X, X_inv

def evaluate_chain_decay(D, X, X_inv, N0, t):
    exp_D = np.diag(np.exp(D * t))
    Nt = X @ exp_D @ X_inv @ N0
    return np.real(Nt) 

# --- FUNÇÕES UTILITÁRIAS ---
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
            txt = f"{item:.4E}" if isinstance(item, (float, int)) else str(item)
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
    
    mode_tab1, mode_tab2 = st.tabs(["Decaimento Simples (A → Estável)", "Decaimento em Cadeia (Matrizes e Gráficos)"])

    with mode_tab1:
        run_simple_mode(chart_theme)

    with mode_tab2:
        run_chain_mode_visual(chart_theme)

def run_simple_mode(chart_theme):
    col_config, col_results = st.columns([1, 2])
    
    with col_config:
        st.subheader("Parâmetros (Simples)")
        
        def update_lambda_callback():
            new_iso = st.session_state.simple_iso
            new_lambda = st.session_state.isotopes[new_iso]["lambda"]
            st.session_state.simple_lam = float(new_lambda)
            st.session_state.iso_escolhido = new_iso

        iso_list = list(st.session_state.isotopes.keys())

        if "iso_escolhido" not in st.session_state:
            st.session_state.iso_escolhido = "Césio-137" if "Césio-137" in iso_list else iso_list[0]

        idx_padrao = 0
        if st.session_state.iso_escolhido in iso_list:
            idx_padrao = iso_list.index(st.session_state.iso_escolhido)
            
        selected_iso = st.selectbox("Isótopo", iso_list, index=idx_padrao, key="simple_iso", on_change=update_lambda_callback)
        iso_data = st.session_state.isotopes[selected_iso]
        
        if "simple_lam" not in st.session_state:
            st.session_state.simple_lam = float(iso_data["lambda"])

        custom_lambda = st.number_input("Lambda (anos⁻¹)", format="%.4E", key="simple_lam")
        
        saved_hl = iso_data.get('half_life', 0)
        saved_unit = iso_data.get('unit', 'anos')
        st.caption(f"Registro Salvo: Meia-vida = {saved_hl:.4E} {saved_unit}")

        st.markdown("---")
        st.markdown("**Tempo de Simulação**")
        
        c1, c2 = st.columns([2, 1])
        t_val = c1.number_input("Duração", value=100.0, key="simple_t", format="%.4E")
        t_unit = c2.selectbox("Unidade", list(CONVERSIONS_TO_YEARS.keys()), index=4, key="simple_unit")
        
        st.markdown("**Qtd Inicial**")
        input_mode = st.radio("Entrada:", ["Massa (g)", "Núcleos (N0)"], horizontal=True, key="simple_mode")
        
        N0 = 0
        atomic_w = iso_data["atomic_weight"]
        
        if input_mode == "Massa (g)":
            mass_initial = st.number_input("Massa (g)", value=1.0, format="%.4E", key="simple_mass")
            N0 = mass_to_nuclei(mass_initial, atomic_w)
        else:
            N0 = st.number_input("N0", value=1.0e20, format="%.4E", key="simple_n0")
            
        steps = st.slider("Passos do Gráfico", 10, 500, 100, key="simple_steps")
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
        st.markdown(f"#### Resultado Final: {res_display:.4E} {unit_label}")
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            color = '#00CC96' if chart_theme == 'plotly_dark' else '#1f77b4'
            hover_txt = [f"t={t:.2f}<br>Qtd={y:.4E} {unit_label}" for t, y in zip(t_plot, y_vals)]
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
            
        df_display = df.copy()
        df_display[f"Tempo ({t_unit})"] = df_display[f"Tempo ({t_unit})"].apply(lambda x: f"{x:.4E}")
        df_display[f"Quantidade ({unit_label})"] = df_display[f"Quantidade ({unit_label})"].apply(lambda x: f"{x:.4E}")
        
        st.dataframe(df_display, use_container_width=True, height=300, hide_index=True)

def run_chain_mode_visual(chart_theme):
    st.markdown("### Simulação de Decaimento em Cadeia")
    
    col_config, col_image = st.columns([1, 1.2])
    
    with col_config:
        st.subheader("1. Recorte da Cadeia")
        
        start_element = st.selectbox("Começar em (Pai):", URANIUM_SERIES_ORDER[:-1], index=0)
        start_idx = URANIUM_SERIES_ORDER.index(start_element)
        
        available_ends = URANIUM_SERIES_ORDER[start_idx+1:]
        end_element = st.selectbox("Terminar em (Filho):", available_ends, index=len(available_ends)-1)
        end_idx = URANIUM_SERIES_ORDER.index(end_element)
        
        cadeia_recortada = URANIUM_SERIES_ORDER[start_idx : end_idx+1]
        st.info(f"Trecho calculado: **{' → '.join(cadeia_recortada)}**")
        
        st.markdown("---")
        st.subheader("2. Tempo de Simulação")
        c_t1, c_t2 = st.columns(2)
        t_val = c_t1.number_input("Duração", value=10.00, min_value=0.00, format="%.4E", key="t_input_chain")
        t_unit = c_t2.selectbox("Unidade", list(CONVERSIONS_TO_YEARS.keys()), index=4, key="unit_input_chain")
        
        steps = st.slider("Passos de Tempo (Gráfico)", 10, 500, 100, key="chain_steps")
        log_scale = st.checkbox("Escala Log (Y)", value=True, key="chain_log")

        st.markdown("---")
        st.subheader("3. Massas Iniciais (g)")
        
        present_isotopes = st.multiselect(
            "Isótopos na amostra inicial:",
            options=cadeia_recortada,
            default=[start_element]
        )
        
        inputs_massa = {}
        for iso in present_isotopes:
            inputs_massa[iso] = st.number_input(
                f"Massa de {iso} (g)", 
                value=100.00 if iso == start_element else 0.00, 
                min_value=0.00, 
                format="%.4E", 
                step=1e-2,
                key=f"m_input_{iso}"
            )
                
        vetor_massas_iniciais = np.array([inputs_massa.get(iso, 0.00) for iso in cadeia_recortada])
        
        st.markdown("---")
        if st.button("Calcular Decaimento", type="primary", use_container_width=True):
            st.session_state['mostrar_resultados_cadeia'] = True

    with col_image:
        st.subheader("Mapa da Série Natural")
        if os.path.exists(CHAIN_IMAGE_FILE):
            st.image(CHAIN_IMAGE_FILE, use_container_width=True)
            
        with st.expander(f"Vetor de Entrada ({len(present_isotopes)} isótopos presentes)"):
            df_m0 = pd.DataFrame({
                "Isótopo": cadeia_recortada, 
                "Massa Inicial (g)": vetor_massas_iniciais
            })
            
            # FILTRO: Mostra APENAS os elementos que o usuário marcou na amostra inicial
            df_m0_display = df_m0[df_m0["Isótopo"].isin(present_isotopes)].copy()
            df_m0_display["Massa Inicial (g)"] = df_m0_display["Massa Inicial (g)"].apply(lambda x: f"{x:.4E}")
            st.dataframe(df_m0_display, hide_index=True, use_container_width=True)

    # --- ÁREA DE RESULTADOS ---
    if st.session_state.get('mostrar_resultados_cadeia', False):
        st.markdown("---")
        st.markdown("## Resultados do Cálculo")
        
        # Movi o seletor para cima, assim ele controla a Tabela e o Gráfico juntos
        st.markdown("#### Exibição dos Dados")
        isotopos_para_plotar = st.multiselect(
            "Escolha quais isótopos visualizar na tabela e no gráfico:",
            options=cadeia_recortada,
            default=present_isotopes 
        )
        
        lambdas_cadeia = [URANIUM_SERIES_DATA[iso]["lambda"] for iso in cadeia_recortada]
        D, X, X_inv = precompute_decay_matrices(lambdas_cadeia)
        tempo_em_anos_max = convert_time_to_years(t_val, t_unit)
        massas_finais = evaluate_chain_decay(D, X, X_inv, vetor_massas_iniciais, tempo_em_anos_max)

        col_tabela, col_grafico = st.columns([1, 2.5])
        
        with col_tabela:
            st.markdown("### Massa Final")
            df_resultado = pd.DataFrame({
                "Isótopo": cadeia_recortada,
                "Massa Final (g)": massas_finais
            })
            
            # FILTRO: Aplica a mesma seleção do gráfico à tabela
            df_resultado_display = df_resultado[df_resultado["Isótopo"].isin(isotopos_para_plotar)].copy()
            df_resultado_display["Massa Final (g)"] = df_resultado_display["Massa Final (g)"].apply(lambda x: f"{x:.4E}")
            st.dataframe(df_resultado_display, hide_index=True, use_container_width=True)

        with col_grafico:
            st.markdown("### Evolução Temporal")
            
            max_t = t_val if t_val > 0 else 100
            t_plot = np.linspace(0, max_t, steps + 1)
            
            historico_massas = []
            for t_inst in t_plot:
                t_anos = convert_time_to_years(t_inst, t_unit)
                massas_inst = evaluate_chain_decay(D, X, X_inv, vetor_massas_iniciais, t_anos)
                historico_massas.append(massas_inst)
                
            historico_massas = np.array(historico_massas)

            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                for i, iso in enumerate(cadeia_recortada):
                    # Só plota se estiver na lista selecionada pelo usuário
                    if iso in isotopos_para_plotar:
                        y_vals = historico_massas[:, i]
                        y_vals[y_vals < 1e-25] = np.nan 
                        
                        fig.add_trace(go.Scatter(
                            x=t_plot, 
                            y=y_vals, 
                            mode='lines', 
                            name=iso,
                            hovertemplate=f"<b>{iso}</b><br>Tempo: %{{x:.2f}}<br>Massa: %{{y:.4E}} g<extra></extra>"
                        ))
                    
                setup_graph_layout(fig, "Decaimento do Recorte", t_unit, "Massa (g)", log_scale, chart_theme, max_t)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Instale 'plotly' para visualizar gráficos.")

def render_manager():
    st.title("Gerenciador de Isótopos")
    st.info("💡 Clique em 'Restaurar Padrões' para carregar a tabela completa se ela estiver incompleta.")
    
    df_iso = pd.DataFrame.from_dict(st.session_state.isotopes, orient='index')
    df_iso.reset_index(inplace=True)
    df_iso.rename(columns={'index': 'Nome', 'lambda': 'Lambda (ano⁻¹)', 'half_life': 'Meia-vida', 'unit': 'Unidade'}, inplace=True)
    
    # FORÇANDO FORMATAÇÃO
    df_iso_display = df_iso.copy()
    df_iso_display["Lambda (ano⁻¹)"] = df_iso_display["Lambda (ano⁻¹)"].apply(lambda x: f"{x:.4E}")
    df_iso_display["Meia-vida"] = df_iso_display["Meia-vida"].apply(lambda x: f"{x:.4E}")

    st.dataframe(df_iso_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Adicionar (Use Ponto .)")
        with st.form("add"):
            name = st.text_input("Nome")
            mass = st.number_input("Massa Atômica", 1.0, format="%.4f")
            hl = st.number_input("Meia-vida (em ANOS)", 1.0, format="%.4E")
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