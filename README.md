# lol-tilt-detector

> **Um detector para quando você está tiltado, para te lembrar de fazer uma pausa e sentir o cheiro das flores.**
> **Saindo do Modo Tilt para o Modo Chill:** Um sistema de Ciência de Dados e Machine Learning que identifica sinais precoces de fadiga cognitiva e frustração (*tilt*) em sessões ranqueadas de *League of Legends* — e convida o jogador a fazer uma pausa na natureza antes que a sequência de derrotas piore.

Este projeto prevê a probabilidade de o jogador de League of Legends estar 'tiltado', isto é, extremamente frustrado e irritado após uma sequência de jogos ruins, sugerindo uma pausa com uma interface agradável e calma.

O modelo funciona com base em engenharia de atributos temporais e conecta-se com a Riot Games Developer Portal.

> Este projeto não é endossado pela Riot Games e não reflete as visões ou opiniões da Riot Games ou de qualquer pessoa oficialmente envolvida na produção ou gerenciamento de propriedades da Riot Games.

> Este projeto segue princípios de privaticade com Anonimização (Todos os identificadores (puuid e match_id) presentes na amostra pública passam por hash criptográfico SHA-256 antes de serem versionados) e Segurança de Chaves (Chaves inseridas na interface web são mantidas estritamente em memória durante a sessão e nunca são gravadas em disco).

Este projeto resolve esse problema e ajuda o jogador a relaxar com uma arquitetura de Data Science de ponta a ponta:
1. **Engenharia de Dados & Coleta em Grafo (BFS):** Extrai telemetria detalhada via Riot API (`Account-v1`, `League-v4` e `Match-v5`) com controle resiliente de *Rate Limit* (`HTTP 429`).
2. **Engenharia de Atributos Temporais sem *Data Leakage*:** Segmenta o histórico em sessões contínuas ($\Delta t \le 15\text{ min}$), calcula *baselines* individuais expansivas e sintetiza janelas móveis ($t-3, t-2, t-1 \rightarrow t$).
3. **Modelagem & Validação Agrupada (`GroupKFold`):** Treina um classificador **LightGBM** regularizado garantindo que partidas de um mesmo invocador nunca vazem entre treino e validação.
4. **Produto Interativo em Estética 8-Bit:** Quando o risco de *tilt* ultrapassa o limiar configurado, o webapp transita automaticamente de *Summoner's Rift* para uma floresta relaxante em pixel art com áudio ambiente, sugerindo uma pausa estratégica.

## 🚀 Como usar
Você pode usar o Demo Mode para testar ou utilizar o seu RiotID para que o detector de tilts monitore ao vivo suas seções de jogatinas.

Você vai precisar instalar o [Git](https://git-scm.com), o [Python](https://www.python.org), e adquirir uma Riot API Key no [Riot Developer Portal](https://developer.riotgames.com).

Após isso, clone o repositório, utilizando o seguinte código no seu terminal: 

```
git clone [https://github.com/SEU-USUARIO/lol-tilt-detector.git](https://github.com/SEU-USUARIO/>lol-tilt-detector.git)
```
Entre na pasta do repositório com
```
cd lol-tilt-detector
```
Ative o ambiente virtual em python
 ```
 python -m venv .venv
 ```
No Windows, utilize:
```
.\.venv\Scripts\Activate.ps1
```
No Linux, utilize:
```
source .venv/bin/activate
```
Instale os pacotes necessários com:
```
pip install -r requirements.txt
```
Execute o Webapp com Streamlit
```
streamlit run app.py
```


## Arquitetura do pipeline
```mermaid
flowchart TD
    A["Riot API (Account-v1 & Match-v5)"] -->|"Coleta assíncrona com Rate Limit"| B["Armazenamento Raw (JSON / Parquet / DuckDB)"]
    B -->|"Agrupamento por PUUID + Ordenação Temporal"| C["Segmentação de Sessões (Δt &lt; 15 min entre partidas)"]
    C -->|"Janelas Deslizantes - Rolling Windows de 3 jogos"| D["Feature Engineering (Delta de mortes, visão, surrenders)"]
    D --> E["Modelo Classificador (LightGBM / XGBoost / CatBoost)"]
```

**Coleta de dados vida Riot API**: conexão via dois roteamentos regionais. Account-v1 (americas): Converte GameName#TagLine em PUUID. Match-v5 (americas): Busca a lista de matchIds (type=ranked para evitar ruído de modos for-fun como ARAM) e os detalhes de cada partida.

**Campos críticos do JSON da Match-v5**: são coletados atributos temporais, de resultado e rendição e indicadores de tilt.

**Segmentação de sessão**: segue uma fórmula baseada na variável ´gameCreation´ em que:

$$\Delta t = \text{gameCreation}_t - \text{gameEndTimestamp}_{t-1}$$

Isto é, partidas com menos de 15 minutos de intervalo de descanso são classificadas como pertencentes à mesma sessão. O modelo é acionado após *4 partidas consecutivas*.

## Variáveis utilizadas na engenharia de atributos temporais

- **Momentum / Fadiga**
    - Feature: loss_streak_session
        - Número de derrotas seguidas nas últimas 3 partidas (0 a 3).
    - Feature: avg_rest_time_sec
        - Média de $\Delta t$ entre as partidas, a partir do clique em 'Jogar novamente'.
    - Feature: session_duration_min
        - Tempo total acumulado jogando na sessão antes da partida $t$.
- **Tendência de morte**
    - Feature: deaths_per_min_slope
        - Inclinação (regressão linear simples) de deaths / min de $t-3$ até $t-1$. Se a taxa de mortes está subindo jogo a jogo, o jogador está jogando de forma cada vez mais imprudente.
- **Tendência de visão**
    - Feature: pink_wards_delta
        - $(\text{Pinks}_{t-1} - \text{Pinks}_{t-3})$ ou razão contra a média histórica do jogador. Jogadores tiltados param de gastar ouro com controle de mapa e focam apenas em itens de dano.
- **Frustração e uso do chat**
    - Feature: spam_pings_ratio
        - Razão de pings de alerta/desaparecido na última partida ($t-1$) vs média do jogador. Jogadores tiltados usam spam de pings para conflitos com a equipe e para expressar frustração, porque é mais rápido do que digitar no chat.
- **Colapso/rendição rápida**
    - Feature: had_surrender_prev
        - Flag binária se a partida $t-1$ acabou em rendição (gameEndedInSurrender). Sair de um jogo que terminou em rendição carrega a frustração para a próxima partida. 
- **Champion swap**
    - Feature: champion_switch_rate
        - Quantos campeões diferentes jogou nas últimas 3 partidas (1 a 3). Trocar muito de campeão após derrotas pode ser uma forma de tilt pick
- **Trol pick**
    - Feature mais complexa, detalhada abaixo:

Calcula-se a popularidade de campeão numa rota:

$$P(\text{Role} \mid \text{Champion}) = \frac{\text{Partidas do Campeão naquela Rota}}{\text{Total de Partidas do Campeão na Base}}$$

Exemplo: Se a Lulu aparece em 94% das vezes como UTILITY (Suporte), 4% no MIDDLE e 0,05% na JUNGLE, qualquer partida dela como JUNGLE recebe uma pontuação próxima de zero.

Diferenciamos um jogador que usa um personagem incomum pela maestria com o campeão, por meio do histórico coletado (player_champ_games).

O sinal real de tilt vem da combinação de is_off_meta_pick == 1 E player_champ_games == 0 (primeira vez jogando com um boneco totalmente fora de posição logo após uma derrota).

Essa análise é complementada pela autofill_or_role_tilt comparada com a rota geralmente jogada pelo jogador em primary_role. Se o jogador é Main ADC (BOTTOM em 80% dos jogos), cai na JUNGLE e escolhe um campeão sem sinergia com a selva, a probabilidade de colapso emocional na partida dispara.

## Estrutura do repositório
```mermaid
flowchart LR
    root["📁 lol-tilt-detector/"]

    root --> data["📁 data/"]
    root --> src["📁 src/"]
    root --> app["📄 app.py<br/><i>Dashboard interativo em Streamlit</i>"]
    root --> setup["📄 setup.py<br/><i>Assistente CLI para criar o .env</i>"]
    root --> env["📄 .env.example<br/><i>Modelo das variáveis de ambiente</i>"]
    root --> gitignore["📄 .gitignore<br/><i>Ignora .env, .venv, data/raw/*</i>"]
    root --> req["📄 requirements.txt<br/><i>Dependências do projeto</i>"]
    root --> readme["📄 README.md<br/><i>Guia de instalação e arquitetura</i>"]

    data --> raw["📁 raw/<br/><i>JSONs/Parquets brutos</i>"]
    data --> sample["📄 sample_matches.parquet<br/><i>Amostra anonimizada para teste</i>"]

    src --> init["📄 __init__.py"]
    src --> collector["📄 collector.py<br/><i>Crawler Riot API com Rate Limit</i>"]
    src --> features["📄 features.py<br/><i>Sessões (&lt;15min), troll picks e deltas</i>"]
    src --> model["📄 model.py<br/><i>Treinamento e predição (LGBM / XGB)</i>"]
```