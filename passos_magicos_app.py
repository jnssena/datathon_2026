import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# Página
st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta baseada no azul e no laranja da associação
TEMA = {
    "azul_escuro": "#0E3A65",
    "azul": "#145089",
    "azul_medio": "#2E6DA8",
    "azul_claro": "#7FA6C9",
    "azul_suave": "#E8EEF5",
    "laranja": "#E87C31",
    "laranja_escuro": "#C4621E",
    "laranja_claro": "#F5B989",
    "laranja_suave": "#FDF0E6",
    "cinza": "#64748B",
}

# Estrutura da página
# Cards e Side bar de todas as abas
st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{font-family: 'Inter', sans-serif !important;}}

/* Fundo cinza */
.stApp {{background: #f2f2f2;}}

/* Sidebar azul escuro */
[data-testid="stSidebar"] {{
    background: {TEMA['azul_escuro']} !important;
    border-right: 1px solid {TEMA['azul']};
}}
[data-testid="stSidebar"] * {{
    color: #eaf1f8 !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {{
    color: {TEMA['azul_claro']} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}}

/* Cards brancos */
.card {{
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}}

/* Card Central (Capa da Pagina) */
.card-hero {{
    background: linear-gradient(135deg, {TEMA['azul_escuro']} 0%, {TEMA['azul_medio']} 100%);
    color: white;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
}}
.card-hero h1, .card-hero p, .card-hero span {{
    color: white !important;
}}

/* Card resultado (Cores diferentes para cada faixa de risco) */
.result-baixo  {{ background: #e8f5e9; border: 2px solid #4caf50; }}
.result-medio  {{ background: {TEMA['laranja_suave']}; border: 2px solid {TEMA['laranja']}; }}
.result-alto   {{ background: #fdecea; border: 2px solid #d93025; }}

/* Card KPIs */
.metric-card {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}}
.metric-value {{
    font-size: 2.2rem;
    font-weight: 600;
    color: {TEMA['azul']};
    line-height: 1;
}}
.metric-label {{
    font-size: 0.75rem;
    color: {TEMA['cinza']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}}

/* Título principal do header */
.hero-title {{
    font-size: 2.1rem;
    font-weight: 600;
    color: white;
    margin: 0;
    line-height: 1.2;
}}
.hero-sub {{
    font-size: 0.9rem;
    color: {TEMA['laranja_claro']};
    margin-top: 6px;
}}

/* Abas */
.stTabs [data-baseweb="tab"] {{
    font-weight: 500;
    color: {TEMA['cinza']};
    border-radius: 8px 8px 0 0;
}}
.stTabs [aria-selected="true"] {{
    color: {TEMA['azul']} !important;
    border-bottom: 2px solid {TEMA['laranja']} !important;
}}

/* Botão da Sidebar */
.stButton > button {{
    background: linear-gradient(135deg, {TEMA['laranja']}, {TEMA['laranja_escuro']}) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(232,124,49,0.4) !important;
}}

/* Oculta elementos padrão do streamlit */
#MainMenu, footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# Carregamento do modelo treinado no notebook
# O cache_resource garante que o arquivo seja lido uma unica vez
@st.cache_resource(show_spinner="Carregando o modelo...")
def carregar_artefato():
    caminho = Path(__file__).parent / "modelos" / "modelo_risco_defasagem.joblib"
    if not caminho.exists():
        st.error(
            f"Arquivo do modelo não encontrado em {caminho}. "
            "Rode o notebook até a célula que salva o artefato."
        )
        st.stop()
    return joblib.load(caminho)


artefato = carregar_artefato()

modelo = artefato["modelo"]
encoders = artefato["encoders"]
ordem_colunas = artefato["ordem_colunas"]
variaveis_categoricas = artefato["variaveis_categoricas"]
corte_padrao = artefato["corte_sugerido"]
metricas = artefato["metricas_teste"]
painel = artefato["painel"]

ULTIMO_ANO = int(painel["ano"].max())
turma_atual = painel[painel["ano"] == ULTIMO_ANO].copy()

DESCRICAO_FASE = {
    0: "ALFA (1º e 2º ano)", 1: "Fase 1 (3º e 4º ano)", 2: "Fase 2 (5º e 6º ano)",
    3: "Fase 3 (7º e 8º ano)", 4: "Fase 4 (9º ano)", 5: "Fase 5 (1º EM)",
    6: "Fase 6 (2º EM)", 7: "Fase 7 (3º EM)", 8: "Fase 8 (Universitários)",
}


def calcular_ian(defasagem):
    """O IAN da base é uma escada: 2,5 / 5 / 10 conforme o nível de defasagem."""
    if defasagem <= -3:
        return 2.5
    if defasagem < 0:
        return 5.0
    return 10.0


def preparar_entrada(dados: dict) -> pd.DataFrame:
    """Aplica os mesmos encoders do treino e devolve as colunas na ordem certa."""
    entrada = pd.DataFrame([dados])
    for coluna in variaveis_categoricas:
        le = encoders[coluna]
        valor = entrada[coluna].astype(str)
        # se aparecer uma categoria nova, cai na primeira conhecida
        valor_seguro = valor.map(lambda v: v if v in le.classes_ else le.classes_[0])
        entrada[coluna] = le.transform(valor_seguro)
    return entrada[ordem_colunas]


def pontuar_turma(df: pd.DataFrame) -> np.ndarray:
    """Calcula a probabilidade de risco para um conjunto de alunos."""
    base = df.copy()
    for coluna in variaveis_categoricas:
        le = encoders[coluna]
        valor = base[coluna].astype(str)
        valor_seguro = valor.map(lambda v: v if v in le.classes_ else le.classes_[0])
        base[coluna] = le.transform(valor_seguro)
    return modelo.predict_proba(base[ordem_colunas])[:, 1]


def faixa_risco(probabilidade, corte):
    if probabilidade >= corte:
        return "Risco alto", "result-alto", "#d93025"
    if probabilidade >= corte * 0.6:
        return "Atenção", "result-medio", TEMA["laranja"]
    return "Risco baixo", "result-baixo", "#2e7d32"


# Configurações dos campos que servirão como input da Sidebar
with st.sidebar:
    st.markdown("### Dados do Aluno")

    with st.expander("Perfil", expanded=True):
        genero_pt = st.selectbox("Gênero", ["Feminino", "Masculino"])
        genero = {"Feminino": "F", "Masculino": "M"}[genero_pt]

        idade = st.slider("Idade", 7, 27, 13)
        ano_ingresso = st.slider("Ano de ingresso na Passos Mágicos",
                                 2016, ULTIMO_ANO, ULTIMO_ANO - 2)
        anos_na_pm = ULTIMO_ANO - ano_ingresso

        instituicao = st.selectbox("Instituição de ensino",
                                   list(encoders["instituicao"].classes_),
                                   index=len(encoders["instituicao"].classes_) - 1)

    with st.expander("Fase", expanded=True):
        fase = st.selectbox("Fase atual", list(DESCRICAO_FASE.keys()),
                            index=3, format_func=lambda f: DESCRICAO_FASE[f])
        fase_ideal = st.selectbox("Fase ideal para a idade", list(DESCRICAO_FASE.keys()),
                                  index=4, format_func=lambda f: DESCRICAO_FASE[f])
        defasagem = fase - fase_ideal
        st.caption(f"Defasagem calculada: **{defasagem:+d}** fase(s)")

    with st.expander("Indicadores", expanded=True):
        ieg = st.slider("IEG — Engajamento", 0.0, 10.0, 7.5, 0.1)
        ida = st.slider("IDA — Aprendizagem", 0.0, 10.0, 6.3, 0.1)
        ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 7.0, 0.1)
        iaa = st.slider("IAA — Autoavaliação", 0.0, 10.0, 8.3, 0.1)
        ips = st.slider("IPS — Psicossocial", 0.0, 10.0, 6.8, 0.1)

    with st.expander("Notas e avaliações", expanded=False):
        nota_mat = st.slider("Nota de Matemática", 0.0, 10.0, 6.2, 0.1)
        nota_por = st.slider("Nota de Português", 0.0, 10.0, 6.2, 0.1)
        nota_ing = st.slider("Nota de Inglês", 0.0, 10.0, 6.5, 0.1)
        qtd_avaliadores = st.slider("Nº de avaliadores", 1, 6, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    calcular = st.button("Calcular Risco")

# O IAN e o INDE não são digitados: derivam do que já foi informado
ian = calcular_ian(defasagem)
PESOS_INDE = {"IAN": 0.10, "IDA": 0.20, "IEG": 0.20, "IAA": 0.10,
              "IPS": 0.10, "IPV": 0.20}
valores_inde = {"IAN": ian, "IDA": ida, "IEG": ieg, "IAA": iaa, "IPS": ips, "IPV": ipv}
inde = sum(valores_inde[k] * p for k, p in PESOS_INDE.items()) / sum(PESOS_INDE.values())

dados_aluno = {
    "idade": float(idade), "anos_na_pm": float(anos_na_pm),
    "fase": float(fase), "fase_ideal": float(fase_ideal),
    "defasagem": float(defasagem), "inde": float(inde),
    "IAA": iaa, "IEG": ieg, "IPS": ips, "IDA": ida, "IPV": ipv, "IAN": ian,
    "nota_mat": nota_mat, "nota_por": nota_por, "nota_ing": nota_ing,
    "qtd_avaliadores": float(qtd_avaliadores),
    "genero": genero, "instituicao": instituicao,
}

# Capa da página
st.markdown(f"""
<div class="card-hero" style="padding:32px 36px">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px">
        <div>
            <h1 class="hero-title">Risco de Defasagem Escolar</h1>
            <p class="hero-sub">Associação Passos Mágicos — modelo {artefato['nome_modelo']}</p>
        </div>
        <div style="display:flex; gap:28px; flex-wrap:wrap">
            <div style="text-align:center">
                <div style="font-size:2rem; font-weight:600; color:white">{metricas['auc']:.2f}</div>
                <div style="font-size:0.7rem; color:{TEMA['laranja_claro']}; letter-spacing:0.1em">AUC</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2rem; font-weight:600; color:white">{len(turma_atual)}</div>
                <div style="font-size:0.7rem; color:{TEMA['laranja_claro']}; letter-spacing:0.1em">ALUNOS EM {ULTIMO_ANO}</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2rem; font-weight:600; color:white">{painel['ano'].nunique()}</div>
                <div style="font-size:0.7rem; color:{TEMA['laranja_claro']}; letter-spacing:0.1em">ANOS DE BASE</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Avaliar Aluno", "Priorizar Turma", "Panorama"])

# Aba Avaliar Aluno
with tab1:
    if calcular:
        entrada = preparar_entrada(dados_aluno)
        probabilidade = float(modelo.predict_proba(entrada)[0, 1])
        rotulo, classe_css, cor = faixa_risco(probabilidade, corte_padrao)

        col_esq, col_dir = st.columns([1.1, 1], gap="large")

        with col_esq:
            st.markdown(f"""
            <div class="card {classe_css}" style="text-align:center">
                <div style="font-size:0.8rem; color:{TEMA['cinza']}; letter-spacing:0.1em;
                            text-transform:uppercase">Probabilidade de entrar em risco</div>
                <div style="font-size:3.6rem; font-weight:600; color:{cor}; line-height:1.1;
                            margin:8px 0">{probabilidade:.0%}</div>
                <div style="font-size:1.1rem; font-weight:600; color:{cor}">{rotulo}</div>
                <div style="font-size:0.8rem; color:{TEMA['cinza']}; margin-top:10px">
                    corte de referência: {corte_padrao:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # KPIs derivados
            k1, k2, k3 = st.columns(3)
            for coluna, valor, rotulo_kpi in [
                (k1, f"{inde:.2f}", "INDE estimado"),
                (k2, f"{defasagem:+d}", "Defasagem"),
                (k3, f"{ian:.1f}", "IAN"),
            ]:
                with coluna:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{valor}</div>
                        <div class="metric-label">{rotulo_kpi}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Quadrante da matriz de priorização
            ja_defasado = defasagem < 0
            risco_alto = probabilidade >= corte_padrao
            if ja_defasado and risco_alto:
                quadrante = "Já defasado · risco alto — atrasado e piorando. Intervenção intensiva."
            elif not ja_defasado and risco_alto:
                quadrante = "Em dia · risco alto — prestes a perder o passo. É aqui que o modelo agrega mais."
            elif ja_defasado:
                quadrante = "Já defasado · risco baixo — atrasado, mas estável. Reforço continuado."
            else:
                quadrante = "Em dia · risco baixo — trajetória saudável. Acompanhamento padrão."

            st.markdown(f"""
            <div class="card">
                <div style="font-weight:600; color:{TEMA['azul']}; margin-bottom:6px">
                    Matriz de priorização</div>
                <div style="color:{TEMA['cinza']}; font-size:0.92rem">{quadrante}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_dir:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-weight:600;font-size:0.95rem;color:{TEMA["azul"]};'
                'margin:0 0 8px 0">Onde este aluno está em relação à turma</p>',
                unsafe_allow_html=True)

            indicadores = ["IEG", "IDA", "IPV", "IAA", "IPS"]
            media_turma = [turma_atual[i].mean() for i in indicadores]
            valores_aluno = [ieg, ida, ipv, iaa, ips]

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            y = np.arange(len(indicadores))
            ax.barh(y + 0.2, media_turma, height=0.38, color=TEMA["azul_claro"],
                    label=f"Média da turma {ULTIMO_ANO}")
            ax.barh(y - 0.2, valores_aluno, height=0.38, color=TEMA["laranja"],
                    label="Este aluno")
            for i, valor in enumerate(valores_aluno):
                ax.text(valor + 0.15, i - 0.2, f"{valor:.1f}", va="center", fontsize=8)
            ax.set_yticks(y)
            ax.set_yticklabels(indicadores)
            ax.set_xlim(0, 11)
            ax.set_xlabel("nota do indicador (0–10)", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.grid(True, linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Card pedindo para preencher a sidebar
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:60px 40px; border:2px dashed #cbd5e1">
            <div style="color:{TEMA['cinza']}; font-size:0.95rem">
                Preencha os dados do aluno na barra lateral<br>
                e clique em <strong>Calcular Risco</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Aba Priorizar Turma
with tab2:
    st.markdown(f"""
    <div class="card">
        <div style="font-weight:600; color:{TEMA['azul']}; margin-bottom:6px">
            Lista de acompanhamento</div>
        <div style="color:{TEMA['cinza']}; font-size:0.92rem">
            Todos os {len(turma_atual)} alunos de {ULTIMO_ANO} ordenados pela probabilidade
            de entrar em defasagem no próximo ciclo. Ajuste a capacidade de atendimento
            para definir quantos entram na lista.
        </div>
    </div>
    """, unsafe_allow_html=True)

    probabilidades = pontuar_turma(turma_atual)
    ranking = turma_atual.assign(probabilidade=probabilidades).sort_values(
        "probabilidade", ascending=False).reset_index(drop=True)

    capacidade = st.slider(
        "Capacidade de atendimento (% da turma)", 5, 50, 25, 5,
        help="Quantos alunos a equipe consegue acompanhar de perto no próximo ciclo",
    )
    quantidade = int(len(ranking) * capacidade / 100)
    selecionados = ranking.head(quantidade)
    corte_efetivo = selecionados["probabilidade"].min()

    k1, k2, k3, k4 = st.columns(4)
    for coluna, valor, rotulo_kpi in [
        (k1, f"{quantidade}", "Alunos priorizados"),
        (k2, f"{corte_efetivo:.0%}", "Corte aplicado"),
        (k3, f"{(selecionados['defasagem'] < 0).mean():.0%}", "Já defasados na lista"),
        (k4, f"{selecionados['inde'].mean():.2f}", "INDE médio da lista"),
    ]:
        with coluna:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{valor}</div>
                <div class="metric-label">{rotulo_kpi}</div>
            </div>
            """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.3, 1], gap="large")

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tabela = selecionados[[
            "id_aluno", "probabilidade", "fase", "fase_ideal", "defasagem",
            "inde", "IEG", "IDA", "IPV",
        ]].copy()
        tabela["probabilidade"] = (tabela["probabilidade"] * 100).round(1)
        tabela = tabela.rename(columns={
            "id_aluno": "Aluno", "probabilidade": "Risco (%)", "fase": "Fase",
            "fase_ideal": "Fase ideal", "defasagem": "Defas.", "inde": "INDE",
        })
        st.dataframe(tabela, hide_index=True, height=420)

        st.download_button(
            "Baixar lista em CSV",
            tabela.to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig"),
            file_name=f"alunos_prioritarios_{ULTIMO_ANO}.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-weight:600;font-size:0.95rem;color:{TEMA["azul"]};'
            'margin:0 0 8px 0">Distribuição do risco na turma</p>',
            unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5.5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.hist(ranking["probabilidade"], bins=30, color=TEMA["azul_claro"],
                edgecolor="white")
        ax.axvline(corte_efetivo, color=TEMA["laranja"], linestyle="--", linewidth=2,
                   label=f"corte {corte_efetivo:.0%}")
        ax.set_xlabel("Probabilidade de risco", fontsize=9)
        ax.set_ylabel("Nº de alunos", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-weight:600;font-size:0.95rem;color:{TEMA["azul"]};'
            'margin:0 0 8px 0">Quadrantes de priorização</p>',
            unsafe_allow_html=True)

        quadros = pd.crosstab(
            ranking["defasagem"].lt(0).map({True: "Já defasado", False: "Em dia"}),
            (ranking["probabilidade"] >= corte_efetivo).map(
                {True: "Risco alto", False: "Risco baixo"}),
        ).reindex(index=["Em dia", "Já defasado"],
                  columns=["Risco baixo", "Risco alto"]).fillna(0).astype(int)

        fig, ax = plt.subplots(figsize=(5.5, 3))
        sns.heatmap(quadros, annot=True, fmt="d", cbar=False, ax=ax,
                    cmap=sns.light_palette(TEMA["laranja"], as_cmap=True),
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

# Aba Panorama
with tab3:
    anos = sorted(painel["ano"].unique())
    ORDEM_PEDRA = ["Quartzo", "Ágata", "Ametista", "Topázio"]
    CORES_PEDRA = [TEMA["azul_claro"], TEMA["azul_medio"],
                   TEMA["laranja_claro"], TEMA["laranja"]]

    # Cards indicadores
    inde_atual = painel.loc[painel["ano"] == ULTIMO_ANO, "inde"].mean()
    inde_inicial = painel.loc[painel["ano"] == anos[0], "inde"].mean()
    pct_defasado = painel.loc[painel["ano"] == ULTIMO_ANO, "defasagem"].lt(0).mean() * 100
    pct_topazio = (painel.loc[painel["ano"] == ULTIMO_ANO, "pedra"]
                   .eq("Topázio").mean() * 100)

    c1, c2, c3, c4 = st.columns(4)
    for coluna, valor, rotulo_kpi in [
        (c1, f"{inde_atual:.2f}", f"INDE médio {ULTIMO_ANO}"),
        (c2, f"{inde_atual - inde_inicial:+.2f}", f"Variação desde {anos[0]}"),
        (c3, f"{pct_defasado:.0f}%", "Alunos defasados"),
        (c4, f"{pct_topazio:.0f}%", "Alunos Topázio"),
    ]:
        with coluna:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{valor}</div>
                <div class="metric-label">{rotulo_kpi}</div>
            </div>
            """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    # Evolução do INDE
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-weight:600;font-size:0.95rem;color:{TEMA["azul"]};'
            'margin:0 0 8px 0">Evolução do INDE médio</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        serie = painel.groupby("ano")["inde"].mean()
        ax.plot(serie.index, serie.values, "o-", color=TEMA["azul"], linewidth=2.5)
        for ano, valor in serie.items():
            ax.annotate(f"{valor:.2f}", (ano, valor), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=TEMA["azul"])
        ax.set_xticks(anos)
        ax.set_ylabel("INDE médio", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Composição por pedra
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-weight:600;font-size:0.95rem;color:{TEMA["azul"]};'
            'margin:0 0 8px 0">Composição por Pedra</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        composicao = (pd.crosstab(painel["ano"], painel["pedra"], normalize="index")
                      .reindex(columns=ORDEM_PEDRA) * 100)
        base = np.zeros(len(composicao))
        for pedra, cor in zip(ORDEM_PEDRA, CORES_PEDRA):
            valores = composicao[pedra].values
            ax.bar(composicao.index.astype(str), valores, bottom=base, color=cor,
                   label=pedra, edgecolor="white", width=0.6)
            for x, (b, v) in enumerate(zip(base, valores)):
                if v > 6:
                    ax.text(x, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            base += valores
        ax.set_ylabel("% dos alunos", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    col_c, col_d = st.columns(2, gap="large")

    # Nível de defasagem por ano
    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-weight:600;font-size:0.95rem;color:{TEMA["azul"]};'
            'margin:0 0 8px 0">Nível de defasagem por ano</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ordem_niveis = ["Severa (3+)", "Alta (2)", "Moderada (1)", "Adequado", "Adiantado"]
        cores_niveis = ["#8c2f14", TEMA["laranja_escuro"], TEMA["laranja_claro"],
                        TEMA["azul_medio"], TEMA["azul_escuro"]]
        distribuicao = (pd.crosstab(painel["ano"], painel["nivel_defasagem"],
                                    normalize="index")
                        .reindex(columns=ordem_niveis).fillna(0) * 100)
        esquerda = np.zeros(len(distribuicao))
        for nivel, cor in zip(ordem_niveis, cores_niveis):
            valores = distribuicao[nivel].values
            ax.barh(distribuicao.index.astype(str), valores, left=esquerda,
                    color=cor, label=nivel)
            esquerda += valores
        ax.set_xlabel("% dos alunos", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Engajamento x aprendizagem
    with col_d:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-weight:600;font-size:0.95rem;color:{TEMA["azul"]};'
            f'margin:0 0 8px 0">Engajamento e aprendizagem em {ULTIMO_ANO}</p>',
            unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        dados = turma_atual.dropna(subset=["IEG", "IDA"])
        ax.scatter(dados["IEG"], dados["IDA"], s=16, alpha=0.45, color=TEMA["azul_medio"])
        coeficientes = np.polyfit(dados["IEG"], dados["IDA"], 1)
        eixo_x = np.linspace(dados["IEG"].min(), dados["IEG"].max(), 50)
        ax.plot(eixo_x, np.polyval(coeficientes, eixo_x), color=TEMA["laranja"],
                linewidth=2.5)
        ax.set_xlabel("IEG — Engajamento", fontsize=9)
        ax.set_ylabel("IDA — Aprendizagem", fontsize=9)
        ax.set_title(f"correlação = {dados['IEG'].corr(dados['IDA']):.2f}",
                     fontsize=9, color=TEMA["cinza"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <div style="font-weight:600; color:{TEMA['azul']}; margin-bottom:6px">
            Sobre o modelo</div>
        <div style="color:{TEMA['cinza']}; font-size:0.9rem; line-height:1.7">
            <strong>Alvo:</strong> {artefato['definicao_alvo']}.<br>
            <strong>Treino:</strong> {artefato['treinado_em']} &nbsp;·&nbsp;
            <strong>Validação:</strong> {artefato['validado_em']}.<br>
            <strong>Desempenho:</strong> AUC {metricas['auc']:.3f} &nbsp;·&nbsp;
            PR-AUC {metricas['pr_auc']:.3f} (aleatório = {metricas['prevalencia']:.3f})
            &nbsp;·&nbsp; Brier {metricas['brier']:.3f}.<br><br>
            Este modelo é um <strong>alerta precoce, não um ranking de gravidade</strong>.
            Alunos já severamente defasados recebem probabilidade baixa por efeito de teto:
            eles quase não têm para onde piorar. Por isso o resultado deve ser lido junto
            com o nível de defasagem atual, como na matriz de quadrantes.
        </div>
    </div>
    """, unsafe_allow_html=True)
