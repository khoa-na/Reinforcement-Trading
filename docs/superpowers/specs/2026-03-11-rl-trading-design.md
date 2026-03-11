# Reinforcement Learning Trading System - Design Specification

**Date:** 2026-03-11
**Project:** RL-based Cryptocurrency Futures Trading System
**Version:** 1.0

---

## 1. Overview

A reinforcement learning trading system that trades USDT-margined futures on the H1 timeframe using PPO algorithm. The system targets 2-5% monthly returns with max 15% drawdown, using 10x leverage with agent-controlled long/short positions.

---

## 2. Goals & Success Criteria

| Metric | Target |
|--------|--------|
| Beat buy-and-hold | Yes |
| Avg monthly return | 2-5% |
| Max drawdown | <15% |
| Sharpe ratio | >1.0 |
| Liquidation count | 0 |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKTEST ENGINE                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  Data    │───▶│    Agent      │───▶│   Portfolio    │  │
│  │  Fetcher │    │   (PPO)       │    │   Manager      │  │
│  └──────────┘    └──────────────┘    └────────────────┘  │
│       │                  │                    │            │
│       ▼                  ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ENVIRONMENT (TradingEnv)               │   │
│  │   - State representation                            │   │
│  │   - Reward calculation                             │   │
│  │   - Position tracking                              │   │
│  │   - Risk controls (SL/TP)                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Data Fetcher** — Fetch OHLCV from Binance
2. **TradingEnv** — Gymnasium environment for RL
3. **Agent (PPO)** — Policy network for action selection
4. **Portfolio Manager** — Track equity, positions, PnL
5. **Backtest Runner** — Run episodes, log results

---

## 4. Market & Assets

- **Exchange:** Binance (USDT-Margined Futures)
- **Timeframe:** H1 (1 hour)
- **Assets:** Top 5-10 by liquidity
  - BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, MATIC
- **Training period:** 60% of data (by time) - chronological, no shuffling
- **Validation period:** 20% of data (by time) - for hyperparameter tuning
- **Test period:** 20% of data (by time) - final evaluation
- **Note:** Time-series data uses temporal split only (no random shuffling)

---

## 5. Action Space

| Action | Behavior |
|--------|----------|
| Open Long 25% | Open long with 25% margin (2.5x leverage effective) |
| Open Long 50% | Open long with 50% margin (5x leverage effective) |
| Open Long 75% | Open long with 75% margin (7.5x leverage effective) |
| Open Long 100% | Open long with 100% margin (10x leverage effective) |
| Close Long | Close existing long position |
| Open Short 25% | Open short with 25% margin (2.5x leverage effective) |
| Open Short 50% | Open short with 50% margin (5x leverage effective) |
| Open Short 75% | Open short with 75% margin (7.5x leverage effective) |
| Open Short 100% | Open short with 100% margin (10x leverage effective) |
| Close Short | Close existing short position |
| Reverse 25% | Close and open opposite with 25% margin |
| Reverse 50% | Close and open opposite with 50% margin |
| Reverse 75% | Close and open opposite with 75% margin |
| Reverse 100% | Close and open opposite with 100% margin |
| Hold | Do nothing |

**Total: 15 actions**

**Rules:**
- Maximum 1 position at a time (long OR short, not both)
- If in a long position: Open Long = Hold, Open Short = Reverse (with selected %)
- If in a short position: Open Short = Hold, Open Long = Reverse (with selected %)
- Margin required: 10% × position_size_% × 10x leverage = position_size_% of equity

---

## 6. State Space (Features)

### Price Features
| Feature | Description |
|---------|-------------|
| returns_1h | 1-hour return |
| returns_4h | 4-hour return |
| returns_24h | 24-hour return |
| volatility_24h | 24-hour rolling volatility |

### Technical Indicators
| Feature | Description |
|---------|-------------|
| rsi_14 | Relative Strength Index (14 periods) |
| macd | MACD line |
| macd_signal | MACD signal line |
| macd_hist | MACD histogram |
| bb_upper | Bollinger Bands upper |
| bb_middle | Bollinger Bands middle |
| bb_lower | Bollinger Bands lower |
| atr_14 | Average True Range |

### Position Features
| Feature | Description |
|---------|-------------|
| in_position | 0 = no position, 1 = long, -1 = short |
| position_pnl_pct | Current position PnL % |
| unrealized_pnl | Unrealized PnL in USDT |

### Market Features
| Feature | Description |
|---------|-------------|
| volume_ratio | Current volume / avg volume |
| high_low_ratio | (high - low) / close |

### Time Features
| Feature | Description |
|---------|-------------|
| hour_sin | Hour of day (sin encoded) |
| hour_cos | Hour of day (cos encoded) |
| day_sin | Day of week (sin encoded) |
| day_cos | Day of week (cos encoded) |

**Total:** ~20 features per timestep

### Feature Normalization

| Method | Application |
|--------|-------------|
| Z-score standardization | Price returns, volatility, ATR |
| Min-max [0,1] | RSI, position PnL % |
| Cyclic encoding | Hour of day, day of week |
| Moving average normalization | Indicators divided by close price |

**Implementation:**
- Compute running statistics (mean, std) during warmup period
- First 24 bars (warmup): use expanding window, clip extreme values
- After warmup: use rolling window (e.g., 100 bars) for stability
- Handle NaN: replace with 0 or default values

### Warmup Period

- **Warmup bars:** 24 bars (1 day of H1 data)
- **Purpose:** Initialize rolling indicator statistics
- **During warmup:**
  - Episode does not start (no actions, no rewards)
  - Accumulate price history for indicator calculation
  - After warmup: indicators have valid values
- **Episode starts** at bar 25 and runs for 168 bars

---

## 7. Reward Function

```
reward = step_return - λ × drawdown_penalty

Where:
- step_return = (current_equity - previous_equity) / initial_equity
- current_equity = balance + unrealized_pnl + realized_pnl
- drawdown_penalty = max(0, current_drawdown)^2 × 0.1
- current_drawdown = (peak_equity - current_equity) / peak_equity
- peak_equity = maximum equity seen in episode so far
- λ = 0.1 (penalty coefficient)
```

**Key design choices:**
- **Step-based reward** (not cumulative) - agent rewarded for change in equity, not for holding
- **Initial equity normalization** - rewards are % of starting capital, not margin
- **Drawdown penalty** - discourages large losses
- This prevents the "hold and wait" exploit where agent opens random positions

---

## 8. Risk Management

| Condition | Action |
|-----------|--------|
| Position loss > 1% of entry price | Auto-close (stop-loss) |
| Position gain > 4% of entry price | Auto-close (take-profit) |
| Margin ratio < 10% (near liquidation) | Auto-close position |
| Total drawdown > 15% | End episode early |

**Liquidation Protection:**
- With 10x leverage, ~10% adverse move = liquidation
- Margin ratio = (margin + unrealized_pnl) / (position_value × maintenance_margin_ratio)
- Close position when margin ratio drops below 10%
- This provides buffer before actual liquidation

**Fee Estimation:**
- Maker fee: 0.02%
- Taker fee: 0.04%
- Slippage: 0.05% (applied to execution price)

---

## 9. Multi-Asset Training

- **Strategy:** Train one agent per asset (separate model for each symbol)
- **Rationale:** Each crypto has different dynamics; single-agent multi-asset is harder to train
- **Execution:** Train separate PPO models for BTC, ETH, etc.
- **Selection:** Use best-performing model for live trading OR ensemble

**Alternative (if compute allows):**
- Train on combined data, randomly sample asset each episode
- Agent learns general market patterns across assets

---

### Order Execution

| Parameter | Value |
|-----------|-------|
| Order type | Market (instant execution) |
| Slippage model | Fixed 0.05% adverse |
| Fill simulation | Immediate at simulated price |

**Note:** Backtest uses market orders for simplicity. For more realistic backtests, consider limit orders with fill probability modeling.

---

## 10. Execution Details

## 11. Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Episode length | 168 bars (1 week of H1) |
| Training episodes | 3000 |
| Learning rate | 3e-4 |
| Batch size | 64 |
| Gamma (discount) | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |

---

## 12. Data Requirements

### OHLCV Fields
- open, high, low, close, volume
- Fetch via Binance API or stored CSV

### Time Range
- Minimum: 1 year of historical data
- Recommended: 2+ years for robust training

---

## 13. Implementation Plan

### Phase 1: Data & Environment
- [ ] Data fetcher for Binance
- [ ] Technical indicator calculator
- [ ] Gymnasium environment (TradingEnv)
- [ ] Reward function implementation

### Phase 2: Agent Training
- [ ] PPO agent implementation
- [ ] Training loop
- [ ] Checkpoint saving

### Phase 3: Backtesting
- [ ] Backtest runner
- [ ] Metrics calculation
- [ ] Results visualization

### Phase 4: Evaluation
- [ ] Train/test split evaluation
- [ ] Compare against buy-and-hold
- [ ] Drawdown analysis

---

## 14. File Structure

```
reinforcement-trading/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py          # Binance data fetching
│   │   └── indicators.py       # Technical indicators
│   ├── env/
│   │   ├── __init__.py
│   │   └── trading_env.py      # Gymnasium environment
│   ├── agent/
│   │   ├── __init__.py
│   │   └── ppo.py              # PPO agent
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── manager.py          # Portfolio tracking
│   └── backtest/
│       ├── __init__.py
│       └── runner.py           # Backtest execution
├── configs/
│   └── config.yaml             # Configuration
├── scripts/
│   └── train.py                # Training script
├── tests/
│   └── ...
├── docs/
│   └── specs/
│       └── 2026-03-11-rl-trading-design.md
├── requirements.txt
└── README.md
```

---

## 15. Notes

- This is a backtesting system first. Live trading requires additional risk controls.
- 10x leverage is aggressive. The risk management layer is critical.
- RL trading systems are notoriously difficult to profit from. This design addresses common failure modes but success is not guaranteed.
- Always forward test on paper trading before any live capital.
