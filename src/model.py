"""
model.py
========
Módulo de Treinamento, Validação Cruzada e Explicabilidade do Detector de Tilt.

Decisões Metodológicas:
- Validação Cruzada via GroupKFold agrupada por 'puuid' (impede data leakage entre
  partidas de um mesmo invocador no treino e na validação).
- Uso de LightGBM regularizado para capturar interações não lineares entre fadiga,
  sequência de derrotas e quedas de telemetria sem sobreajuste (overfitting).
- Exportação do artefato final (modelo + metadados + importância das features)
  para consumo pelo dashboard interativo (app.py).
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# Lista explícita das features comportamentais e temporais usadas no modelo
FEATURE_COLUMNS = [
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

# Dicionário de nomes legíveis para exibição no terminal e no dashboard
FEATURE_LABELS = {
    "game_index_in_session": "Nº do Jogo na Sessão",
    "losses_last_3": "Derrotas nas Últimas 3 Partidas",
    "lost_prev_game": "Perdeu a Partida Anterior",
    "avg_rest_min_last_3": "Tempo Médio de Descanso (min)",
    "session_minutes_played": "Tempo Acumulado na Sessão (min)",
    "deaths_pm_trend": "Tendência de Mortes/min (t-1 vs t-3)",
    "deaths_vs_baseline_prev": "Mortes/min vs Média Histórica",
    "kda_vs_baseline_prev": "KDA vs Média Histórica",
    "pink_wards_trend": "Tendência de Pink Wards",
    "pinks_last_3_avg": "Média de Pink Wards (Últimos 3 Jogos)",
    "surrenders_last_3": "Rendições nas Últimas 3 Partidas",
    "troll_signals_last_3": "Sinais de Troll Pick (Últimos 3 Jogos)",
    "pings_pm_prev": "Pings de Frustração/min (Jogo Anterior)",
}


class TiltModelTrainer:
    """
    Classe responsável por validar, treinar e serializar o classificador de tilt.
    """

    def __init__(self, alert_threshold: float = 0.65, n_splits: int = 5):
        """
        Args:
            alert_threshold (float): Limiar de probabilidade acima do qual o sistema
                recomenda uma pausa ao jogador (prioriza precisão para evitar falsos alertas).
            n_splits (int): Número de divisões na validação cruzada GroupKFold.
        """
        self.alert_threshold = alert_threshold
        self.n_splits = n_splits
        self.model = self._build_estimator()

    @staticmethod
    def _build_estimator() -> lgb.LGBMClassifier:
        """
        Instancia o LightGBM com hiperparâmetros conservadores para evitar overfitting
        em dados comportamentais ruidosos.
        """
        return lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

    def evaluate_cv(self, df: pd.DataFrame) -> dict:
        """
        Executa validação cruzada agrupada por jogador (puuid) e retorna
        as métricas consolidadas fora da amostra (Out-of-Fold).
        """
        X = df[FEATURE_COLUMNS]
        y = df["target_loss"]
        groups = df["puuid"]

        unique_players = groups.nunique()
        actual_splits = min(self.n_splits, unique_players)

        if actual_splits < 2:
            raise ValueError(
                "❌ É necessário ter pelo menos 2 jogadores distintos com sessões válidas "
                "para executar a validação cruzada."
            )

        gkf = GroupKFold(n_splits=actual_splits)
        oof_preds = np.zeros(len(df))

        logger.info(f"🧪 Executando GroupKFold ({actual_splits} folds) sobre {len(df)} sequências de {unique_players} jogadores...")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            clf = self._build_estimator()
            clf.fit(X_train, y_train)
            oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]

        # Calcula métricas globais Out-of-Fold
        roc_auc = roc_auc_score(y, oof_preds)
        pr_auc = average_precision_score(y, oof_preds)

        # Métricas no limiar de recomendação de pausa (alta confiança)
        y_pred_alert = (oof_preds >= self.alert_threshold).astype(int)
        alert_count = int(y_pred_alert.sum())

        if alert_count > 0:
            precision_alert = precision_score(y, y_pred_alert, zero_division=0)
            recall_alert = recall_score(y, y_pred_alert, zero_division=0)
        else:
            precision_alert = 0.0
            recall_alert = 0.0

        metrics = {
            "total_samples": len(df),
            "unique_players": unique_players,
            "base_loss_rate": round(float(y.mean()), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "alert_threshold": self.alert_threshold,
            "alerts_triggered": alert_count,
            "precision_at_threshold": round(float(precision_alert), 4),
            "recall_at_threshold": round(float(recall_alert), 4),
        }
        return metrics

    def fit_and_save(self, df: pd.DataFrame, model_path: str = "models/tilt_model.joblib") -> dict:
        """
        Avalia o modelo via CV, treina a versão final com 100% dos dados disponíveis
        e salva o artefato completo no disco.
        """
        metrics = self.evaluate_cv(df)

        X = df[FEATURE_COLUMNS]
        y = df["target_loss"]

        logger.info("🏋️ Treinando modelo final com toda a base processada...")
        self.model.fit(X, y)

        # Extrai importância das features (ganho de informação)
        importances = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "label": [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        artifact = {
            "model": self.model,
            "feature_columns": FEATURE_COLUMNS,
            "feature_labels": FEATURE_LABELS,
            "alert_threshold": self.alert_threshold,
            "metrics": metrics,
            "feature_importances": importances.to_dict(orient="records")
        }

        out_file = Path(model_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, out_file)

        logger.info(f"✅ Modelo salvo com sucesso em '{out_file}'.")
        return artifact


# ==============================================================================
# EXECUÇÃO DIRETA PELO TERMINAL
# ==============================================================================
if __name__ == "__main__":
    features_path = Path("data/processed/features.parquet")

    if not features_path.exists():
        raise FileNotFoundError(
            f"❌ Arquivo '{features_path}' não encontrado. Rode 'python src/features.py' primeiro."
        )

    df_features = pd.read_parquet(features_path)

    if len(df_features) < 20:
        logger.warning(
            f"⚠️ Atenção: Você tem apenas {len(df_features)} sequências válidas no dataset. "
            "Para métricas estatísticas mais estáveis, colete mais jogadores no collector.py."
        )

    trainer = TiltModelTrainer(alert_threshold=0.65, n_splits=5)
    artifact = trainer.fit_and_save(df_features)

    metrics = artifact["metrics"]
    print("\n" + "=" * 55)
    print("📊 RELATÓRIO DE VALIDAÇÃO (OUT-OF-FOLD / GROUP-KFOLD)")
    print("=" * 55)
    print(f"• Sequências avaliadas : {metrics['total_samples']} (de {metrics['unique_players']} jogadores)")
    print(f"• Taxa base de derrota : {metrics['base_loss_rate']:.1%}")
    print(f"• ROC-AUC              : {metrics['roc_auc']:.4f}")
    print(f"• PR-AUC               : {metrics['pr_auc']:.4f}")
    print(f"• Limiar de Alerta     : >= {metrics['alert_threshold']:.0%} de risco")
    print(f"• Precisão no Alerta   : {metrics['precision_at_threshold']:.1%} ({metrics['alerts_triggered']} alertas disparados)")
    print("-" * 55)
    print("🏆 TOP 5 SINAIS COMPORTAMENTAIS MAIS RELEVANTES:")
    df_imp = pd.DataFrame(artifact["feature_importances"]).head(5)
    for _, row in df_imp.iterrows():
        print(f"  - {row['label']:<40} (Score: {row['importance']})")
    print("=" * 55 + "\n")