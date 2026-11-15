"""
app.py
======
Webapp Interativo 8-Bit: Detector de Tilt no League of Legends.

Configurado para operação com Personal API Key (via st.secrets ou .env),
cache de consultas ao vivo (TTL 10 min) para proteção de Rate Limit e
conformidade visual com os termos de desenvolvedor da Riot Games.
"""

import base64
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.collector import RiotCollector
from src.features import TiltFeatureEngineer
from src.model import TiltModelTrainer, FEATURE_COLUMNS

# ==============================================================================
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Detector de Tilts | 8-Bit Edition",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()


# ==============================================================================
# FUNÇÕES AUXILIARES DE MÍDIA E ESTILO 8-BIT
# ==============================================================================
def get_base64_file(file_path: str) -> str:
    """Converte uma imagem ou áudio local em Base64."""
    path = Path(file_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_riot_api_key() -> str:
    """Recupera a Personal API Key dos Secrets do Streamlit Cloud ou do .env local."""
    try:
        secret_key = st.secrets.get("RIOT_API_KEY", "")
        if secret_key:
            return secret_key
    except Exception:
        pass

    env_key = os.getenv("RIOT_API_KEY", "")
    return "" if "sua-chave" in env_key else env_key


def apply_8bit_theme(is_tilted: bool):
    """Aplica o plano de fundo centralizado, botões pixel-art e contraste."""
    bg_file = "assets/forest_8bit.png" if is_tilted else "assets/lol8bit.png"
    b64_img = get_base64_file(bg_file)

    if b64_img:
        bg_css = f"""
        background-image: linear-gradient(rgba(10, 15, 20, 0.42), rgba(10, 15, 20, 0.55)),
                          url("data:image/png;base64,{b64_img}") !important;
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        """
    elif is_tilted:
        bg_css = "background: linear-gradient(180deg, #1b4332 0%, #2d6a4f 50%, #081c15 100%) !important;"
    else:
        bg_css = "background: linear-gradient(135deg, #1a2a4a 0%, #244b3c 50%, #1d3557 100%) !important;"

    border_color = "#52b788" if is_tilted else "#f4a261"
    box_bg = "rgba(13, 40, 24, 0.92)" if is_tilted else "rgba(17, 24, 39, 0.92)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

        /* Oculta completamente a sidebar e o header padrão transparente */
        [data-testid="stSidebar"], [data-testid="collapsedControl"], header[data-testid="stHeader"] {{
            display: none !important;
        }}

        .block-container {{
            padding-top: 2.2rem !important;
            padding-bottom: 3.5rem !important;
        }}

        .stApp {{
            {bg_css}
        }}

        /* Container centralizado estilo caixa de diálogo 8-bit */
        .rpg-dialog-box {{
            background-color: {box_bg};
            border: 4px solid {border_color};
            box-shadow: 0 0 0 4px #000000, 8px 8px 0px rgba(0, 0, 0, 0.8);
            padding: 24px 28px;
            border-radius: 2px;
            max-width: 650px;
            margin: 15px auto 22px auto;
            text-align: center;
        }}

        .pixel-title {{
            font-family: 'Press Start 2P', monospace;
            color: #ffd166;
            text-shadow: 3px 3px 0px #000000;
            font-size: 1.18rem;
            line-height: 1.6;
            margin-bottom: 14px;
        }}

        .forest-title {{
            font-family: 'Press Start 2P', monospace;
            color: #b7e4c7;
            text-shadow: 3px 3px 0px #081c15;
            font-size: 1.18rem;
            line-height: 1.6;
            margin-bottom: 14px;
        }}

        .pixel-text {{
            font-family: 'VT323', monospace;
            color: #f1faee;
            font-size: 1.38rem;
            line-height: 1.4;
            text-shadow: 2px 2px 0px #000000;
        }}

        /* Rótulo legível em alto contraste acima do Selectbox e Inputs */
        .pixel-field-label {{
            font-family: 'VT323', monospace;
            font-size: 1.4rem;
            color: #ffd166;
            background-color: rgba(17, 24, 39, 0.92);
            border: 3px solid {border_color};
            border-bottom: none;
            padding: 6px 14px;
            display: block;
            text-shadow: 2px 2px 0px #000000;
            box-shadow: 4px 0px 0px #000000;
            margin-bottom: -14px;
        }}

        /* Estilização do Selectbox e Text Input */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {{
            background-color: #111827 !important;
            border: 3px solid {border_color} !important;
            border-radius: 0px !important;
            color: #f8fafc !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.25rem !important;
            box-shadow: 4px 4px 0px #000000 !important;
        }}

        /* Botões em estilo Arcade / 8-Bit */
        .stButton > button {{
            font-family: 'Press Start 2P', monospace !important;
            font-size: 0.68rem !important;
            line-height: 1.4 !important;
            color: #111827 !important;
            background-color: #ffd166 !important;
            border: 3px solid #000000 !important;
            border-radius: 0px !important;
            box-shadow: 4px 4px 0px #000000 !important;
            padding: 12px 10px !important;
            transition: transform 0.05s ease, box-shadow 0.05s ease !important;
            margin-top: 6px !important;
        }}

        .stButton > button:hover {{
            background-color: #ffe194 !important;
            color: #000000 !important;
            transform: translate(-1px, -1px) !important;
            box-shadow: 5px 5px 0px #000000 !important;
        }}

        .stButton > button:active {{
            transform: translate(3px, 3px) !important;
            box-shadow: 1px 1px 0px #000000 !important;
        }}

        /* Cards de métricas */
        [data-testid="stMetric"] {{
            background-color: rgba(15, 23, 42, 0.92);
            border: 3px solid {border_color};
            padding: 12px;
            box-shadow: 4px 4px 0px #000000;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'VT323', monospace !important;
            font-size: 1.25rem !important;
            color: #e2e8f0 !important;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'Press Start 2P', monospace !important;
            font-size: 1.05rem !important;
            color: #ffd166 !important;
        }}

        /* Player de áudio discreto fixado no canto inferior direito */
        .discreet-audio-player {{
            position: fixed;
            bottom: 16px;
            right: 16px;
            z-index: 9999;
            background-color: rgba(13, 40, 24, 0.94);
            border: 2px solid #74c69d;
            padding: 6px 10px;
            border-radius: 4px;
            box-shadow: 4px 4px 0px #000000;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .discreet-audio-player span {{
            font-family: 'VT323', monospace;
            color: #b7e4c7;
            font-size: 1.1rem;
        }}
        .discreet-audio-player audio {{
            height: 26px;
            width: 190px;
            opacity: 0.85;
        }}

        /* Aviso legal discreto da Riot Games no rodapé */
        .riot-disclaimer {{
            max-width: 650px;
            margin: 35px auto 0 auto;
            padding: 8px 12px;
            background-color: rgba(10, 15, 25, 0.78);
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-family: sans-serif;
            font-size: 0.72rem;
            color: #cbd5e1;
            text-align: center;
            line-height: 1.35;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_discreet_forest_audio():
    """Renderiza um player de som compacto e discreto no canto inferior direito."""
    b64_wav = get_base64_file("assets/forest.wav")
    if b64_wav:
        st.markdown(
            f"""
            <div class="discreet-audio-player">
                <span>🌿 Som Ambiente</span>
                <audio controls autoplay loop>
                    <source src="data:audio/wav;base64,{b64_wav}" type="audio/wav">
                </audio>
            </div>
            """,
            unsafe_allow_html=True,
        )


def style_8bit_chart(fig, title: str):
    """Aplica identidade visual arcade/8-bit aos gráficos Plotly."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="VT323, monospace", size=24, color="#ffd166"),
        ),
        font=dict(family="VT323, monospace", size=18, color="#f1faee"),
        paper_bgcolor="rgba(15, 23, 42, 0.92)",
        plot_bgcolor="rgba(10, 15, 28, 0.94)",
        margin=dict(l=30, r=30, t=55, b=30),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)", zeroline=False),
    )
    fig.update_traces(marker_line_color="#000000", marker_line_width=2)
    return fig


# ==============================================================================
# CARREGAMENTO DO MODELO, CACHE DA RIOT API E PREDIÇÃO
# ==============================================================================
@st.cache_resource
def load_or_train_model():
    model_path = Path("models/tilt_model.joblib")
    if model_path.exists():
        return joblib.load(model_path)

    sample_path = Path("data/sample_matches.parquet")
    if sample_path.exists():
        df_raw = pd.read_parquet(sample_path)
        engineer = TiltFeatureEngineer()
        df_feat = engineer.build_features(df_raw)
        trainer = TiltModelTrainer()
        return trainer.fit_and_save(df_feat)

    return None


@st.cache_data
def load_sample_features():
    sample_path = Path("data/sample_matches.parquet")
    if not sample_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    df_raw = pd.read_parquet(sample_path)
    engineer = TiltFeatureEngineer()
    df_feat = engineer.build_features(df_raw)
    return df_raw, df_feat


@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_player_matches(game_name: str, tag_line: str, region: str) -> pd.DataFrame:
    """
    Consulta as últimas 8 partidas ranqueadas de um jogador usando a Personal API Key.
    Cacheado por 10 minutos (ttl=600) para proteger o Rate Limit da chave.
    """
    api_key = get_riot_api_key()
    if not api_key:
        raise ValueError("Chave da Riot API não configurada no servidor.")

    collector = RiotCollector(api_key=api_key, region=region)
    puuid = collector.get_puuid_by_riot_id(game_name, tag_line)
    if not puuid:
        return pd.DataFrame()

    match_ids = collector.get_ranked_match_ids(puuid, count=8)
    rows = []
    for m_id in match_ids:
        m_data = collector.get_match_details(m_id)
        if m_data:
            parsed = collector.parse_match_participants(m_data)
            rows.extend([r for r in parsed if r["puuid"] == puuid])

    return pd.DataFrame(rows)


def analyze_player_session(df_player_raw: pd.DataFrame, artifact: dict):
    engineer = TiltFeatureEngineer(max_session_gap_min=15.0)
    df_feat = engineer.build_features(df_player_raw)

    if df_feat.empty:
        return None

    latest_row = df_feat.sort_values("game_start_ms").iloc[[-1]]
    model = artifact["model"]
    prob_loss = float(model.predict_proba(latest_row[FEATURE_COLUMNS])[0][1])

    return {
        "prob_loss": prob_loss,
        "latest_features": latest_row.iloc[0].to_dict(),
    }


# ==============================================================================
# ESTADO DA SESSÃO E CARREGAMENTO INICIAL
# ==============================================================================
if "is_tilted" not in st.session_state:
    st.session_state.is_tilted = False
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"  # 'home' | 'live' | 'analytics'
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

artifact = load_or_train_model()
if not artifact:
    st.error("❌ Modelo ou amostra não encontrados. Rode `src/collector.py` e `src/model.py` primeiro.")
    st.stop()

threshold = artifact["alert_threshold"]
df_sample_raw, df_sample_feat = load_sample_features()

apply_8bit_theme(is_tilted=st.session_state.is_tilted)

# ==============================================================================
# ESTADO 1: MODO CHILL ATIVADO (TELA DA FLORESTA QUANDO DETECTA TILT)
# ==============================================================================
if st.session_state.is_tilted:
    st.markdown(
        """
        <div class="rpg-dialog-box">
            <div class="forest-title">🌿 MODO CHILL ATIVADO 🌿</div>
            <div class="pixel-text">
                "Parece que a última partida te deixou um pouco tiltado, certo?<br>
                Que tal fazer uma pausa e apreciar a natureza?<br>
                <strong>Respire fundo, relaxe!</strong>"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_discreet_forest_audio()

    if st.session_state.analysis_result:
        feat = st.session_state.analysis_result["latest_features"]
        prob = st.session_state.analysis_result["prob_loss"]
        _, c1, c2, c3, _ = st.columns([1.5, 2, 2, 2, 1.5])
        c1.metric("Risco de Tilt", f"{prob:.1%}")
        c2.metric("Derrotas Seguidas", f"{int(feat['losses_last_3'])} / 3")
        c3.metric("Pausa Entre Jogos", f"{feat['avg_rest_min_last_3']:.1f} min")

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([2.2, 2.2, 2.2])
    if col_btn.button("⚔️ Voltar para Summoner's Rift", use_container_width=True):
        st.session_state.is_tilted = False
        st.session_state.current_view = "home"
        st.rerun()

# ==============================================================================
# ESTADO 2: TELA INICIAL (SUMMONER'S RIFT 8-BIT + GRADE DE BOTÕES)
# ==============================================================================
elif st.session_state.current_view == "home":
    st.markdown(
        """
        <div class="rpg-dialog-box">
            <div class="pixel-title">⚔️ Bem vindo ao Detector de Tilts! ⚔️</div>
            <div class="pixel-text">
                Se detectarmos sinais de fadiga ou frustração, vamos te ajudar a sair do modo tilt para o modo chill.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_mid, _ = st.columns([1.5, 3, 1.5])
    with col_mid:
        if not df_sample_feat.empty:
            players_available = df_sample_feat["puuid"].unique().tolist()
            st.markdown(
                '<span class="pixel-field-label">👤 Selecione um Invocador (Amostra Anonimizada):</span>',
                unsafe_allow_html=True,
            )
            selected_player = st.selectbox(
                "Selecione um Invocador (Amostra Anonimizada):",
                players_available,
                label_visibility="collapsed",
            )

            # Linha 1 de botões 8-bit
            b1, b2 = st.columns(2)
            if b1.button("🔍 Verificar Estado Mental", use_container_width=True):
                df_p = df_sample_raw[df_sample_raw["puuid"] == selected_player].sort_values("game_start_ms")
                res = analyze_player_session(df_p, artifact)
                st.session_state.analysis_result = res
                st.session_state.is_tilted = bool(res and res["prob_loss"] >= threshold)
                st.rerun()

            if b2.button("🌲 Simular Alerta de Tilt", use_container_width=True):
                st.session_state.is_tilted = True
                st.session_state.analysis_result = {
                    "prob_loss": 0.78,
                    "latest_features": {
                        "losses_last_3": 3,
                        "avg_rest_min_last_3": 0.8,
                        "game_index_in_session": 4,
                    },
                }
                st.rerun()

        # Linha 2 de botões 8-bit
        b3, b4 = st.columns(2)
        if b3.button("🔴 Consulta Ao Vivo", use_container_width=True):
            st.session_state.current_view = "live"
            st.rerun()

        if b4.button("📊 Análises 8-Bit", use_container_width=True):
            st.session_state.current_view = "analytics"
            st.rerun()

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        feat = res["latest_features"]
        prob = res["prob_loss"]

        st.markdown("<br>", unsafe_allow_html=True)
        _, c1, c2, c3, c4, _ = st.columns([1, 2, 2, 2, 2, 1])
        c1.metric("Risco Previsto", f"{prob:.1%}")
        c2.metric("Status Mental", "ESTÁVEL ✅")
        c3.metric("Derrotas (Últimos 3)", f"{int(feat['losses_last_3'])} / 3")
        c4.metric("Descanso Médio", f"{feat['avg_rest_min_last_3']:.1f} min")

    st.markdown(
        """
        <div class="riot-disclaimer">
            Detector de Tilts não é endossado pela Riot Games e não reflete as visões ou opiniões da Riot Games
            ou de qualquer pessoa oficialmente envolvida na produção ou gerenciamento das propriedades da Riot Games.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# ESTADO 3: CONSULTA AO VIVO VIA RIOT API (USANDO PERSONAL API KEY)
# ==============================================================================
elif st.session_state.current_view == "live":
    st.markdown(
        """
        <div class="rpg-dialog-box">
            <div class="pixel-title" style="font-size: 1.05rem;">🔴 CONSULTA AO VIVO (RIOT API)</div>
            <div class="pixel-text">
                Digite seu Riot ID (ex: Nick#BR1) para avaliar sua sessão ranqueada recente em tempo real.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_live, _ = st.columns([1.5, 3, 1.5])
    with col_live:
        c_reg, c_id = st.columns([1, 2])
        with c_reg:
            st.markdown('<span class="pixel-field-label">🌎 Região:</span>', unsafe_allow_html=True)
            region_input = st.selectbox(
                "Região",
                ["americas", "europe", "asia", "sea"],
                label_visibility="collapsed",
            )
        with c_id:
            st.markdown('<span class="pixel-field-label">🎮 Riot ID (Nick#TAG):</span>', unsafe_allow_html=True)
            riot_id_query = st.text_input(
                "Riot ID",
                placeholder="Ex: Nick#BR1",
                label_visibility="collapsed",
            )

        btn_l1, btn_l2 = st.columns(2)
        if btn_l1.button("🔍 Analisar Agora", use_container_width=True):
            if "#" not in riot_id_query:
                st.warning("⚠️ Formato inválido! Use 'Nome#TAG'.")
            else:
                game_name, tag_line = [p.strip() for p in riot_id_query.split("#", 1)]
                with st.spinner(f"Consultando partidas recentes de {game_name}#{tag_line}..."):
                    try:
                        df_live = fetch_live_player_matches(game_name, tag_line, region_input)
                        if df_live.empty:
                            st.error("❌ Invocador não encontrado ou sem partidas ranqueadas recentes.")
                        elif len(df_live) < 4:
                            st.info("ℹ️ Invocador possui menos de 4 partidas ranqueadas recentes para análise de sessão.")
                        else:
                            res = analyze_player_session(df_live, artifact)
                            if not res:
                                st.info(
                                    "ℹ️ Nenhuma sessão contínua de pelo menos 4 partidas "
                                    "com menos de 15 min de intervalo foi encontrada (sem sinal de tilt agudo)."
                                )
                            else:
                                st.session_state.analysis_result = res
                                st.session_state.is_tilted = bool(res["prob_loss"] >= threshold)
                                if st.session_state.is_tilted:
                                    st.rerun()
                                else:
                                    st.success(f"✅ Estado Estável! Risco de derrota previsto: {res['prob_loss']:.1%}")
                    except Exception as e:
                        st.error(f"Erro na consulta: {e}")

        if btn_l2.button("⬅️ Voltar ao Início", use_container_width=True):
            st.session_state.current_view = "home"
            st.rerun()

# ==============================================================================
# ESTADO 4: ANÁLISES 8-BIT
# ==============================================================================
elif st.session_state.current_view == "analytics":
    st.markdown(
        """
        <div class="rpg-dialog-box" style="padding: 16px;">
            <div class="pixel-title" style="font-size: 1.05rem; margin-bottom: 6px;">👾 LABORATÓRIO COMPORTAMENTAL 8-BIT 👾</div>
            <div class="pixel-text" style="font-size: 1.2rem;">
                Padrões extraídos das sessões ranqueadas e pesos do modelo LightGBM.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_back, _ = st.columns([2.5, 2, 2.5])
    if col_back.button("⬅️ Voltar ao Início", use_container_width=True):
        st.session_state.current_view = "home"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        df_imp = pd.DataFrame(artifact["feature_importances"]).head(8).sort_values("importance", ascending=True)
        fig_imp = px.bar(
            df_imp,
            x="importance",
            y="label",
            orientation="h",
            color_discrete_sequence=["#ffd166"],
            labels={"importance": "Score de Importância (Splits)", "label": ""},
        )
        fig_imp = style_8bit_chart(fig_imp, "🏆 TOP SINAIS DE TILT (LIGHTGBM)")
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_g2:
        if not df_sample_feat.empty:
            df_snowball = (
                df_sample_feat.groupby("losses_last_3", as_index=False)
                .agg(taxa_derrota=("target_loss", "mean"), qtd=("target_loss", "count"))
            )
            df_snowball["taxa_derrota_pct"] = (df_snowball["taxa_derrota"] * 100).round(1)
            df_snowball["losses_last_3"] = df_snowball["losses_last_3"].astype(int).astype(str) + " derrota(s)"

            fig_snow = px.bar(
                df_snowball,
                x="losses_last_3",
                y="taxa_derrota_pct",
                text="taxa_derrota_pct",
                color="taxa_derrota_pct",
                color_continuous_scale=["#52b788", "#ffd166", "#e63946"],
                labels={"losses_last_3": "Histórico Imediato (t-3..t-1)", "taxa_derrota_pct": "Taxa de Derrota (%)"},
            )
            fig_snow.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_snow.update_layout(coloraxis_showscale=False, yaxis_range=[0, 100])
            fig_snow = style_8bit_chart(fig_snow, "❄️ EFEITO BOLA DE NEVE NA SESSÃO")
            st.plotly_chart(fig_snow, use_container_width=True)

    if not df_sample_feat.empty:
        df_rest = df_sample_feat.copy()
        df_rest["faixa_descanso"] = pd.cut(
            df_rest["avg_rest_min_last_3"],
            bins=[-0.1, 1.0, 3.0, 7.0, 15.0],
            labels=["0 - 1 min (Instantâneo)", "1 - 3 min (Rápido)", "3 - 7 min (Moderado)", "7 - 15 min (Calmo)"],
        )
        df_rest_summary = (
            df_rest.groupby("faixa_descanso", observed=False, as_index=False)["target_loss"]
            .mean()
        )
        df_rest_summary["taxa_derrota_pct"] = (df_rest_summary["target_loss"] * 100).round(1)

        fig_rest = px.bar(
            df_rest_summary,
            x="faixa_descanso",
            y="taxa_derrota_pct",
            text="taxa_derrota_pct",
            color_discrete_sequence=["#4cc9f0"],
            labels={"faixa_descanso": "Intervalo Médio Entre Partidas", "taxa_derrota_pct": "Taxa de Derrota (%)"},
        )
        fig_rest.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_rest.update_layout(yaxis_range=[0, 100])
        fig_rest = style_8bit_chart(fig_rest, "⏱️ IMPACTO DO RE-QUEUE IMPULSIVO")
        st.plotly_chart(fig_rest, use_container_width=True)