"""
features.py
===========
Módulo de Engenharia de Atributos Temporais e Comportamentais para Detecção de Tilt.

Responsabilidades:
1. Detectar escolhas fora do meta / troll picks empiricamente a partir da base.
2. Segmentar o histórico de cada jogador em sessões contínuas (gap <= 15 minutos).
3. Calcular a baseline histórica individual sem data leakage (médias expansivas).
4. Extrair métricas das últimas 3 partidas (t-3, t-2, t-1) para prever o desfecho
   da partida atual (t).
5. Exportar o dataset processado em formato Parquet.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class TiltFeatureEngineer:
    """
    Pipeline de transformação de dados brutos de partidas em features
    sequenciais de fadiga e frustração (tilt).
    """

    def __init__(self, max_session_gap_min: float = 15.0, off_meta_threshold: float = 0.02):
        """
        Args:
            max_session_gap_min (float): Intervalo máximo em minutos entre duas partidas
                para serem consideradas parte da mesma sessão contínua.
            off_meta_threshold (float): Frequência mínima esperada de um campeão em uma
                rota para ser considerado escolha padrão (padrão: 2%).
        """
        self.max_session_gap_min = max_session_gap_min
        self.off_meta_threshold = off_meta_threshold

    def _add_troll_pick_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica picks fora do meta (ex: Lulu Jungle) e anomalias de feitiços.
        Combina a raridade do par (Campeão, Rota) com a falta de experiência prévia
        do invocador com aquele campeão.
        """
        df = df.copy()

        # Ignora partidas onde a API não identificou a rota (ex: modos antigos)
        valid_pos = df["team_position"].isin(["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"])

        champ_total = df.groupby("champion")["match_id"].transform("count")
        champ_role_total = df.groupby(["champion", "team_position"])["match_id"].transform("count")

        # Probabilidade empírica P(Rota | Campeão)
        df["champ_role_prob"] = np.where(valid_pos, champ_role_total / champ_total, 1.0)

        # Marca como off-meta se a presença na rota for < 2% (com mínimo de 10 jogos do campeão na base)
        df["is_off_meta"] = (
            valid_pos
            & (df["champ_role_prob"] < self.off_meta_threshold)
            & (champ_total >= 10)
        ).astype(int)

        # Experiência prévia do jogador com este campeão (ordem estritamente cronológica)
        df = df.sort_values(["puuid", "game_start_ms"]).reset_index(drop=True)
        df["player_prev_champ_games"] = df.groupby(["puuid", "champion"]).cumcount()

        # Troll Pick Flag: Pick fora do meta + Primeira vez jogando com o campeão na base
        df["troll_pick_flag"] = (
            (df["is_off_meta"] == 1) & (df["player_prev_champ_games"] == 0)
        ).astype(int)

        # Anomalia de Feitiço: Caçador (JUNGLE) sem Golpear (Smite = ID 11)
        has_smite = (df["spell1_id"] == 11) | (df["spell2_id"] == 11)
        df["jungle_no_smite_flag"] = (
            (df["team_position"] == "JUNGLE") & (~has_smite)
        ).astype(int)

        # Consolida qualquer sinal grave de sabotagem/troll na seleção
        df["any_troll_signal"] = (
            (df["troll_pick_flag"] == 1) | (df["jungle_no_smite_flag"] == 1)
        ).astype(int)

        return df

    def _segment_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrupa partidas consecutivas de cada jogador em sessões com base no
        tempo de descanso entre o fim de um jogo e o início do próximo.
        """
        df = df.sort_values(["puuid", "game_start_ms"]).reset_index(drop=True)

        # Timestamp de término da partida anterior do mesmo jogador
        df["prev_game_end_ms"] = df.groupby("puuid")["game_end_ms"].shift(1)

        # Intervalo de descanso em minutos entre partidas consecutivas
        df["gap_min"] = (df["game_start_ms"] - df["prev_game_end_ms"]) / (1000.0 * 60.0)
        # Corrige eventuais sobreposições de relógio para 0
        df["gap_min"] = df["gap_min"].clip(lower=0.0)

        # Inicia nova sessão se for o 1º jogo registrado ou se o intervalo > 15 min
        df["is_new_session"] = (df["gap_min"].isna()) | (df["gap_min"] > self.max_session_gap_min)
        df["session_id"] = df.groupby("puuid")["is_new_session"].cumsum()

        # Índice da partida dentro da sessão atual (0 = 1º jogo, 1 = 2º jogo, etc.)
        df["game_index_in_session"] = df.groupby(["puuid", "session_id"]).cumcount()

        return df

    def _add_player_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula a média histórica de desempenho de cada jogador utilizando apenas
        partidas estritamente anteriores à partida atual (shift(1) + expanding()),
        garantindo ausência total de data leakage.
        """
        df = df.sort_values(["puuid", "game_start_ms"]).reset_index(drop=True)
        g_player = df.groupby("puuid")

        # Baseline individual (mínimo de 3 jogos prévios para estabilizar a média)
        df["baseline_deaths_pm"] = g_player["deaths_per_min"].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean()
        )
        df["baseline_kda"] = g_player["kda"].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean()
        )
        df["baseline_pink_wards"] = g_player["pink_wards"].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean()
        )

        # Preenche eventuais NaNs iniciais com a mediana global para não perder amostras
        df["baseline_deaths_pm"] = df["baseline_deaths_pm"].fillna(df["deaths_per_min"].median())
        df["baseline_kda"] = df["baseline_kda"].fillna(df["kda"].median())
        df["baseline_pink_wards"] = df["baseline_pink_wards"].fillna(df["pink_wards"].median())

        return df

    def build_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Executa todo o pipeline de transformação e retorna o dataset pronto
        para treinamento/predição (1 linha = 1 partida alvo t com histórico t-3..t-1).
        """
        if df_raw.empty:
            raise ValueError("❌ O DataFrame de entrada está vazio.")

        logger.info(f"📥 Iniciando engenharia de atributos sobre {len(df_raw)} registros...")

        # 1. Remove remakes precoces caso tenham passado
        df = df_raw[df_raw["early_surrender"] == 0].copy()

        # 2. Adiciona flags de troll picks e sessões
        df = self._add_troll_pick_signals(df)
        df = self._segment_sessions(df)
        df = self._add_player_baselines(df)

        # Total de pings de frustração por minuto na partida
        df["frustration_pings_pm"] = (
            df["enemy_missing_pings"] + df["danger_pings"] + df["bait_pings"]
        ) / df["duration_min"]

        # 3. Extração das janelas das últimas 3 partidas DENTRO da mesma sessão
        g_sess = df.groupby(["puuid", "session_id"])

        # --- Momentum e Fadiga ---
        # Número de derrotas nas últimas 3 partidas (0 a 3)
        df["losses_last_3"] = 3 - (
            g_sess["win"].shift(1) + g_sess["win"].shift(2) + g_sess["win"].shift(3)
        )
        # Flag se perdeu especificamente o jogo imediatamente anterior (t-1)
        df["lost_prev_game"] = 1 - g_sess["win"].shift(1)

        # Tempo médio de descanso entre os últimos 3 jogos (cliques impulsivos em "Jogar Novamente")
        df["avg_rest_min_last_3"] = (
            g_sess["gap_min"].shift(1) + g_sess["gap_min"].shift(2)
        ) / 2.0

        # Tempo acumulado jogando na sessão antes da partida atual (fadiga)
        df["session_minutes_played"] = (
            g_sess["duration_min"].shift(1).fillna(0)
            + g_sess["duration_min"].shift(2).fillna(0)
            + g_sess["duration_min"].shift(3).fillna(0)
        )

        # --- Tendência de Queda de Desempenho ---
        # Inclinação de mortes/min: diferença entre o jogo t-1 e o jogo t-3
        df["deaths_pm_trend"] = g_sess["deaths_per_min"].shift(1) - g_sess["deaths_per_min"].shift(3)

        # Razão de mortes no último jogo comparada à média histórica do próprio jogador
        df["deaths_vs_baseline_prev"] = (
            g_sess["deaths_per_min"].shift(1) / (df["baseline_deaths_pm"] + 1e-5)
        )

        # Queda de KDA no último jogo em relação à própria baseline
        df["kda_vs_baseline_prev"] = (
            g_sess["kda"].shift(1) / (df["baseline_kda"] + 1e-5)
        )

        # --- Negligência de Visão ---
        # Diferença de Pink Wards compradas entre o jogo t-1 e t-3
        df["pink_wards_trend"] = g_sess["pink_wards"].shift(1) - g_sess["pink_wards"].shift(3)
        df["pinks_last_3_avg"] = (
            g_sess["pink_wards"].shift(1)
            + g_sess["pink_wards"].shift(2)
            + g_sess["pink_wards"].shift(3)
        ) / 3.0

        # --- Sinais Agudos de Frustração e Sabotagem ---
        df["surrenders_last_3"] = (
            g_sess["surrender"].shift(1)
            + g_sess["surrender"].shift(2)
            + g_sess["surrender"].shift(3)
        )
        df["troll_signals_last_3"] = (
            g_sess["any_troll_signal"].shift(1)
            + g_sess["any_troll_signal"].shift(2)
            + g_sess["any_troll_signal"].shift(3)
        )
        df["pings_pm_prev"] = g_sess["frustration_pings_pm"].shift(1)

        # --- Variável Alvo (Target Y) ---
        # 1 = Perdeu a 4ª partida (ou subsequente) da sessão | 0 = Venceu
        df["target_loss"] = 1 - df["win"]

        # 4. Filtra apenas partidas que possuem histórico completo de 3 jogos na sessão (índice >= 3)
        feature_cols = [
            "game_index_in_session",
            "losses_last_3",
            "lost_prev_game",
            "avg_rest_min_last_3",
            "session_minutes_played",
            "deaths_pm_trend",
            "deaths_vs_baseline_prev",
            "kda_vs_baseline_prev",
            "pink_wards_trend",
            "pinks_last_3_avg",
            "surrenders_last_3",
            "troll_signals_last_3",
            "pings_pm_prev",
        ]

        id_cols = ["match_id", "puuid", "session_id", "game_start_ms", "champion", "team_position"]
        final_df = df[df["game_index_in_session"] >= 3][id_cols + feature_cols + ["target_loss"]].dropna().copy()

        logger.info(
            f"✅ Engenharia concluída! {len(final_df)} sequências válidas extraídas "
            f"(Taxa de derrota alvo: {final_df['target_loss'].mean():.1%})."
        )
        return final_df


# ==============================================================================
# EXECUÇÃO DIRETA PELO TERMINAL
# ==============================================================================
if __name__ == "__main__":
    raw_path = Path("data/raw/matches_raw.parquet")
    out_path = Path("data/processed/features.parquet")

    if not raw_path.exists():
        raise FileNotFoundError(
            f"❌ Arquivo '{raw_path}' não encontrado. Rode 'python src/collector.py' primeiro."
        )

    df_raw = pd.read_parquet(raw_path)
    engineer = TiltFeatureEngineer(max_session_gap_min=15.0)
    df_features = engineer.build_features(df_raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out_path, index=False)

    print(f"\n💾 Dataset salvo em '{out_path}' ({len(df_features)} linhas).")
    if not df_features.empty:
        print("\nAmostra das features geradas:")
        print(df_features[["losses_last_3", "avg_rest_min_last_3", "deaths_pm_trend", "troll_signals_last_3", "target_loss"]].head(10))