"""
anonymize_sample.py
===================
Gera uma amostra anonimizada a partir dos dados brutos coletados localmente
para ser versionada no GitHub (data/sample_matches.parquet).

Garante conformidade com a Política de Desenvolvedores da Riot Games ao
substituir identificadores reais (PUUID e Match ID) por hashes SHA-256.
"""

import hashlib
from pathlib import Path
import pandas as pd


def hash_id(value: str) -> str:
    """Converte um identificador original em um hash curto irreversível."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def create_anonymized_sample(
    raw_path: str = "data/raw/matches_raw.parquet",
    sample_path: str = "data/sample_matches.parquet",
    max_rows: int = 3000,
) -> None:
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"❌ Arquivo '{raw_file}' não encontrado. Rode o collector.py antes.")

    df = pd.read_parquet(raw_file).head(max_rows).copy()

    # Anonimiza PUUID e Match ID mantendo a integridade relacional (mesmo jogador = mesmo hash)
    df["puuid"] = df["puuid"].apply(lambda x: f"player_{hash_id(x)}")
    df["match_id"] = df["match_id"].apply(lambda x: f"BR1_{hash_id(x)}")

    out_file = Path(sample_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)

    print(f"✅ Amostra anonimizada salva em '{out_file}' ({len(df)} registros, {df['puuid'].nunique()} jogadores).")


if __name__ == "__main__":
    create_anonymized_sample()