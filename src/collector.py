"""
collector.py
============
Módulo responsável pela coleta resiliente de dados de partidas ranqueadas
da API oficial da Riot Games (Endpoints: Account-v1, League-v4 e Match-v5).

Funcionalidades principais:
- Conversão de Riot ID (GameName#TagLine) para PUUID global ou busca automática
  de jogadores ativos no Challenger.
- Controle automático de Rate Limit (HTTP 429) respeitando o header 'Retry-After'.
- Coleta em grafo (BFS - Breadth-First Search) a partir de um jogador semente,
  permitindo escalar a base de dados para centenas de invocadores.
- Extração de telemetria relevante para detecção de tilt (mortes, visão, pings,
  rendições, feitiços e posição jogada).
- Armazenamento eficiente em formato colunar Parquet.
"""

import os
import time
import logging
from pathlib import Path
from collections import deque
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class RiotCollector:
    """
    Cliente para comunicação com a Riot Games API com suporte a rate limiting
    e extração estruturada de partidas.
    """

    def __init__(self, api_key: Optional[str] = None, region: str = "americas"):
        load_dotenv()
        self.api_key = api_key or os.getenv("RIOT_API_KEY")
        self.region = region or os.getenv("RIOT_REGION", "americas")

        if not self.api_key or "sua-chave-aqui" in self.api_key:
            raise ValueError(
                "❌ RIOT_API_KEY não encontrada ou inválida! "
                "Configure o arquivo .env ou passe a chave diretamente."
            )

        self.headers = {"X-Riot-Token": self.api_key}
        self.base_url = f"https://{self.region}.api.riotgames.com"
        self.request_delay = 1.25  # Evita HTTP 429 na chave de desenvolvimento

    def _make_request(self, url: str, params: Optional[dict] = None) -> Optional[dict | list]:
        """Executa uma requisição GET com tratamento automático de Rate Limit (429)."""
        while True:
            time.sleep(self.request_delay)
            response = requests.get(url, headers=self.headers, params=params, timeout=15)

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning(f"⏳ Rate limit atingido (429). Aguardando {retry_after}s...")
                time.sleep(retry_after + 1)
                continue

            elif response.status_code == 404:
                logger.warning(f"⚠️ Recurso não encontrado (404): {url}")
                return None

            elif response.status_code in (401, 403):
                raise PermissionError(
                    "🔑 Chave da Riot expirada ou sem permissão (401/403). "
                    "Gere uma nova chave em https://developer.riotgames.com/"
                )

            elif response.status_code >= 500:
                logger.warning(f"🔧 Erro interno na Riot ({response.status_code}). Tentando em 5s...")
                time.sleep(5)
                continue

            else:
                logger.error(f"❌ Erro inesperado {response.status_code}: {response.text}")
                return None

    def get_puuid_by_riot_id(self, game_name: str, tag_line: str) -> Optional[str]:
        """Converte um Riot ID (ex: 'Nick', 'BR1') no identificador único global (PUUID)."""
        url = f"{self.base_url}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        data = self._make_request(url)
        return data.get("puuid") if data else None

    def get_active_challenger_puuid(self, platform: str = "BR1") -> Optional[str]:
        """Busca automaticamente o PUUID de um jogador ativo no topo do Challenger."""
        url = f"https://{platform.lower()}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
        data = self._make_request(url)

        if not data or not data.get("entries"):
            return None

        top_player = data["entries"][0]
        if "puuid" in top_player:
            return top_player["puuid"]

        summoner_id = top_player.get("summonerId")
        if summoner_id:
            sum_url = f"https://{platform.lower()}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
            sum_data = self._make_request(sum_url)
            return sum_data.get("puuid") if sum_data else None

        return None

    def get_ranked_match_ids(self, puuid: str, count: int = 50) -> list[str]:
        """Retorna a lista de IDs das últimas partidas ranqueadas de um jogador."""
        url = f"{self.base_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {
            "type": "ranked",
            "start": 0,
            "count": min(count, 100)
        }
        data = self._make_request(url, params=params)
        return data if isinstance(data, list) else []

    def get_match_details(self, match_id: str) -> Optional[dict]:
        """Baixa o JSON completo com a telemetria de uma partida específica."""
        url = f"{self.base_url}/lol/match/v5/matches/{match_id}"
        return self._make_request(url)

    @staticmethod
    def parse_match_participants(match_data: dict) -> list[dict]:
        """Extrai as métricas relevantes de TODOS os 10 participantes de uma partida."""
        info = match_data.get("info", {})
        metadata = match_data.get("metadata", {})
        match_id = metadata.get("matchId")

        duration_sec = info.get("gameDuration", 0)
        if duration_sec < 300:  # Ignora remakes (< 5 min)
            return []

        duration_min = duration_sec / 60.0
        game_start = info.get("gameCreation")
        game_end = info.get("gameEndTimestamp") or (game_start + (duration_sec * 1000))
        queue_id = info.get("queueId")

        rows = []
        for p in info.get("participants", []):
            puuid = p.get("puuid")
            if not puuid or puuid == "BOT":
                continue

            kills = p.get("kills", 0)
            deaths = p.get("deaths", 0)
            assists = p.get("assists", 0)
            total_cs = p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)

            rows.append({
                "match_id": match_id,
                "puuid": puuid,
                "queue_id": queue_id,
                "game_start_ms": game_start,
                "game_end_ms": game_end,
                "duration_min": round(duration_min, 2),
                "win": int(p.get("win", False)),
                "surrender": int(p.get("gameEndedInSurrender", False)),
                "early_surrender": int(p.get("gameEndedInEarlySurrender", False)),
                "champion": p.get("championName"),
                "team_position": p.get("teamPosition") or "UNKNOWN",
                "spell1_id": p.get("summoner1Id"),
                "spell2_id": p.get("summoner2Id"),
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "deaths_per_min": round(deaths / duration_min, 4),
                "kda": round((kills + assists) / max(1, deaths), 2),
                "cs_per_min": round(total_cs / duration_min, 2),
                "gold_per_min": round(p.get("goldEarned", 0) / duration_min, 2),
                "pink_wards": p.get("visionWardsBoughtInGame", 0),
                "wards_placed": p.get("wardsPlaced", 0),
                "vision_score_per_min": round(p.get("visionScore", 0) / duration_min, 2),
                "enemy_missing_pings": p.get("enemyMissingPings", 0),
                "danger_pings": p.get("dangerPings", 0),
                "bait_pings": p.get("baitPings", 0),
            })

        return rows

    def crawl_from_seed(
        self,
        seed_game_name: Optional[str] = None,
        seed_tag_line: Optional[str] = None,
        seed_puuid: Optional[str] = None,
        max_players: int = 15,
        matches_per_player: int = 40,
        output_path: str = "data/raw/matches_raw.parquet"
    ) -> pd.DataFrame:
        """Realiza a coleta em largura (BFS) partindo de um Riot ID ou PUUID."""
        if not seed_puuid:
            logger.info(f"🚀 Iniciando coleta a partir de {seed_game_name}#{seed_tag_line}...")
            seed_puuid = self.get_puuid_by_riot_id(seed_game_name, seed_tag_line)

        if not seed_puuid:
            raise ValueError(f"Jogador {seed_game_name}#{seed_tag_line} não encontrado.")

        players_queue = deque([seed_puuid])
        visited_players = set()
        processed_matches = set()
        all_rows = []

        while players_queue and len(visited_players) < max_players:
            current_puuid = players_queue.popleft()
            if current_puuid in visited_players:
                continue

            visited_players.add(current_puuid)
            logger.info(
                f"👤 Jogador {len(visited_players)}/{max_players} | "
                f"Buscando últimas {matches_per_player} partidas..."
            )

            match_ids = self.get_ranked_match_ids(current_puuid, count=matches_per_player)

            for match_id in match_ids:
                if match_id in processed_matches:
                    continue

                processed_matches.add(match_id)
                match_data = self.get_match_details(match_id)

                if not match_data:
                    continue

                parsed_rows = self.parse_match_participants(match_data)
                all_rows.extend(parsed_rows)

                if len(visited_players) + len(players_queue) < max_players * 2:
                    for row in parsed_rows:
                        p_id = row["puuid"]
                        if p_id not in visited_players:
                            players_queue.append(p_id)

            logger.info(
                f"📊 Progresso parcial: {len(processed_matches)} partidas únicas | "
                f"{len(all_rows)} registros coletados."
            )

        if not all_rows:
            logger.warning("⚠️ Nenhuma partida ranqueada válida encontrada.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).drop_duplicates(subset=["match_id", "puuid"])

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_file, index=False)

        logger.info(f"✅ Coleta finalizada! {len(df)} linhas salvas em '{out_file}'.")
        return df


# ==============================================================================
# EXECUÇÃO INTERATIVA PELO TERMINAL
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("🎮 COLETOR DE DADOS - DETECTOR DE TILT (LOL)")
    print("=" * 55)

    riot_id = input(
        "👉 Digite o Riot ID (ex: Nick#BR1) ou aperte ENTER para buscar\n"
        "   automaticamente um jogador ativo do Challenger BR1: "
    ).strip()

    region_input = input("🌎 Região [americas / europe / asia / sea] (Enter = americas): ").strip().lower()
    region = region_input if region_input else "americas"

    players_input = input("👥 Quantos jogadores explorar na rede? (Enter = 10): ").strip()
    max_players = int(players_input) if players_input.isdigit() else 10

    matches_input = input("⚔️  Quantas partidas recentes por jogador? (Enter = 30): ").strip()
    matches_per_player = int(matches_input) if matches_input.isdigit() else 30

    print("-" * 55 + "\n")

    collector = RiotCollector(region=region)

    if not riot_id:
        logger.info("🔍 Buscando um jogador ativo no Challenger BR1...")
        seed_puuid = collector.get_active_challenger_puuid(platform="BR1")
        if not seed_puuid:
            raise RuntimeError("❌ Não foi possível obter um jogador do Challenger. Verifique sua RIOT_API_KEY.")
        
        logger.info(f"✅ Jogador ativo encontrado! PUUID: {seed_puuid[:12]}...")
        df_coletado = collector.crawl_from_seed(
            seed_puuid=seed_puuid,
            max_players=max_players,
            matches_per_player=matches_per_player
        )
    else:
        if "#" not in riot_id:
            raise ValueError("❌ Formato inválido! Use 'Nome#TAG'.")
        game_name, tag_line = [part.strip() for part in riot_id.split("#", 1)]
        df_coletado = collector.crawl_from_seed(
            seed_game_name=game_name,
            seed_tag_line=tag_line,
            max_players=max_players,
            matches_per_player=matches_per_player
        )

    if not df_coletado.empty:
        print("\n📊 Resumo dos dados coletados:")
        print(df_coletado[["match_id", "champion", "team_position", "win", "deaths_per_min"]].head(10))