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
- **Training period:** 70% of data (by time)
- **Test period:** 30% of data (by time)

---

## 5. Action Space

| Action | Behavior |
|--------|----------|
| Open Long | Open long position with 10x leverage |
| Close Long | Close existing long position |
| Open Short | Open short position with 10x leverage |
| Close Short | Close existing short position |
| Hold | Do nothing |

**Rules:**
- Maximum 1 position at a time (long OR short, not both)
- Cannot open new position while one exists
- 10x leverage = 10% margin required

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

---

## 7. Reward Function

```
reward = portfolio_return - λ × drawdown_penalty

Where:
- portfolio_return = (unrealized_pnl + realized_pnl) / margin
- drawdown_penalty = max(0, current_drawdown)^2 × 0.1
- λ = 0.1 (penalty coefficient)
```

**Key design choices:**
- Smoothed rewards reduce noise from bar-to-bar chop
- Drawdown penalty discourages large losses
- Margin-based returns normalize across price levels

---

## 8. Risk Management

| Condition | Action |
|-----------|--------|
| Position loss > 1% | Auto-close (stop-loss) |
| Position gain > 4% | Auto-close (take-profit) |
| Price reaches 90% of liquidation | Auto-close |
| Total drawdown > 15% | End episode early |

**Fee Estimation:**
- Maker fee: 0.02%
- Taker fee: 0.04%
- Slippage: 0.05%

---

## 9. Training Configuration

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

## 10. Data Requirements

### OHLCV Fields
- open, high, low, close, volume
- Fetch via Binance API or stored CSV

### Time Range
- Minimum: 1 year of historical data
- Recommended: 2+ years for robust training

---

## 11. Implementation Plan

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

## 12. File Structure

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

## 13. Notes

- This is a backtesting system first. Live trading requires additional risk controls.
- 10x leverage is aggressive. The risk management layer is critical.
- RL trading systems are notoriously difficult to profit from. This design addresses common failure modes but success is not guaranteed.
- Always forward test on paper trading before any live capital.
