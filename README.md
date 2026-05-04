# stock-simulator

> **A DQN-based stock trading agent. Built primarily to answer one question: do macroeconomic features actually help an RL trader, or do we just feel better including them? Spoiler: the effect is real but not statistically significant.**

A reinforcement-learning trading sandbox. The agent (a DQN, with an enhanced v2 variant using multi-head attention over the asset universe) gets a Gymnasium environment that wraps ~400 US equities with ~25 years of daily price history, plus optional macro indicators. It learns a buy / hold / sell policy episode by episode. Code from this project was used in an accepted paper.

## The actual question

There's a lot of casual "we added GDP / interest rates / VIX as features and our model improved" in trading-RL papers. I wanted to do the ablation honestly — same architecture, same training budget, same random seed, only difference being the macro features.

### The result

| Configuration | Mean portfolio (start: $50k) | Sharpe | Success rate |
|---|---|---|---|
| Minimal model (no macros) | $79,573 ± $35,247 | 0.84 | 92.5% |
| Full model (with macros)  | $85,936 ± $31,774 | 1.13 | 100.0% |

Macro features improved Sharpe by ~35% and lifted success rate to 100%. **But:** t-test p-value on portfolio value = 0.4050. Mann-Whitney p = 0.0791. Mean improvement of 8% sits comfortably inside one standard deviation of either group.

So the answer is: macros *probably* help, the direction is consistent, the statistical evidence is suggestive at best with 40 episodes per arm. To claim significance you'd need to scale episode count by an order of magnitude — which is the version of the experiment I'd run if I had more compute.

The full ablation summary lives in `results.yaml`; per-episode metrics in `stats/v1/`.

## What's in here

```
models/
  ├── v1/
  │   ├── stock_gym.py        Gymnasium env: 400 tickers, daily steps, $50k starting cash
  │   ├── dqn.py              Vanilla DQN with experience replay
  │   └── trade_agent.py      Training loop, epsilon-greedy, target-network sync
  ├── v2/
  │   ├── enhanced_gym.py     Same env + macro features as observation channel
  │   ├── enhanced_dqn.py     DQN with multi-head attention over the asset slice
  │   └── enhanced_agent.py   v1 trainer adapted for the larger state space
  └── ablation.py             Runs both models, dumps the comparison summary
data/
  ├── merged/                 Per-ticker CSVs (Date + OHLCV), already cleaned
  ├── normalized_date_macro.csv  Macro feature panel aligned to trading dates
  └── scripts/                Data fetch / merge / buy-and-hold baseline
dashboard.py                  Streamlit UI for inspecting trained policy on a single stock
saved_models/                 Trained checkpoints + ablation visualizations
stats/                        Per-iteration portfolio JSON + analysis plots
results.yaml                  Headline ablation summary (the table above)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch gymnasium pandas numpy plotly streamlit tqdm

# Reproduce the ablation
python models/ablation.py

# Inspect a trained policy interactively
streamlit run dashboard.py
```

You'll need ~6 GB of RAM for the full env (the asset matrix is dense). A GPU is nice but not required — the network is small. Training one ablation arm (40 episodes, 400 stocks) takes ~25 min on a 3060.

## Caveats I want to be upfront about

- **Backtest, not deployment.** No transaction costs modelled, no slippage, no realistic order execution latency. A live version of this agent would lose money on costs alone.
- **Survivorship bias.** The 400-ticker universe is "stocks that exist today and have ≥25 years of history." Companies that delisted are missing entirely.
- **Not financial advice.** I built this to answer an experimental question, not to manage capital. Use accordingly.
