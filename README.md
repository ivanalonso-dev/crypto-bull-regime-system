# crypto-bull-regime-system

Crypto portfolio trading system that runs **only when BTC is in a bull regime**, as defined by project [006 · btc-regime-detector](https://github.com/ivanalonso-dev/btc-regime-detector). Developed and validated with pre-registered rules, including the parts that failed.

> **TL;DR:** A daily Donchian-breakout trend system (system A) has a **real but modest edge** when BTC is in a strong bull regime: +1.15R per trade on the 9 development coins and **+0.41R on 10 coins never used in development** (positive in 9 of 10). A mean-reversion component (system B) had **no edge** and was removed. Pyramiding looked great in development and **failed out of sample**. On risk-adjusted terms the system does **not** beat simply holding BTC during bull regimes (project 006); its value is a low-drawdown, low-exposure return stream. Live paper trading started on 2026-10-05.

## Final system (v2 — official)

| | Rule |
|---|---|
| **Regime gate** | Trade only when project 006 says *Bull – strong*; close everything when it says *Not bull* |
| **Entry** | Daily close > highest close of the previous 20 days |
| **Initial stop** | 3 × ATR(20) below entry |
| **Exits** | Chandelier trailing stop (highest close − 3 × ATR) · close < lowest close of previous 10 days · regime turns *Not bull* |
| **Sizing** | 0.5% of equity at risk per trade · ≤ 20% per coin · ≤ 100% total · no leverage |
| **Execution** | Signal at close, fill at next open, stops as resting orders, 0.10% fee per side |
| **Universe (live)** | BTC ETH BNB XRP ADA SOL DOGE LINK AVAX + LTC BCH TRX DOT ATOM ETC XLM NEAR FIL UNI (Binance daily USDT candles) |

**Challenger (h2):** same system with a looser exit (4 × ATR trailing, 20-day low). Tracked live in parallel.

## How it was built (and what failed)

### Round 1 — pre-registered v1 (system A trend + system B mean reversion)

Development coins (BTC ETH BNB XRP ADA SOL DOGE LINK AVAX), in-sample 2017–2022, out-of-sample 2023–2026.

| | OOS CAGR | OOS max DD | OOS Calmar |
|---|---|---|---|
| **v1 (A + B)** | 10.3% | −10.7% | 0.96 |
| Buy & hold BTC | 55.1% | −53.0% | 1.04 |
| 006 alone (BTC in bull regime) | 35.3% | −33.1% | 1.06 |
| Equal-weight basket in bull regime | 36.1% | −51.1% | 0.71 |

**Result: 3 / 5 criteria → not validated.** Failed "beat all benchmarks" (0.96 vs 1.06) and "positive expectancy in each sub-system":

| Sub-system | Trades | Expectancy (IS / OOS) | Profit factor (IS / OOS) |
|---|---|---|---|
| A · trend | 188 | +1.56R / +0.79R | 7.2 / 4.1 |
| B · mean reversion | 286 | +0.01R / −0.02R | 1.04 / 0.90 |

System A was positive on all 9 coins and in all 9 neighbouring parameter sets; system B was noise → **v2 = A only**.

### Round 2 — improvement hypotheses, validated on 10 unseen coins

Hypotheses fixed in advance: **H1** pyramiding (up to 3 units, +1 ATR steps) · **H2** looser exit (4 × ATR, 20-day low) · **H3** relative strength (top 3 by 90-day return). Rule: a hypothesis passes development if it improves both Calmar and expectancy; the candidate (all passing hypotheses) is adopted only if it beats v2 on **10 coins never used** (LTC BCH TRX DOT ATOM ETC XLM NEAR FIL UNI).

| Variant (risk 0.5%) | Dev Calmar | Dev R/trade | **Unseen Calmar** | **Unseen max DD** | **Unseen R/trade (ex-top 5)** |
|---|---|---|---|---|---|
| v2 (A only) | 1.29 | 1.15 | **0.38** | −13.4% | 0.41 (0.15) |
| H1 pyramiding | 1.43 | 2.55 | 0.28 | −28.8% | 0.70 (0.09) |
| H2 looser exit | 1.85 | 2.58 | 0.39 | −19.4% | 0.88 (0.35) |
| H3 relative strength | 0.85 | 0.75 | 0.31 | −7.1% | 0.34 (−0.05) |
| Candidate H1 + H2 | 2.04 | 5.98 | 0.33 | −36.2% | 1.35 (0.35) |
| *006 alone (same period)* | — | — | *0.83* | *−45.4%* | — |

- **Candidate not adopted** (Calmar 0.33 < 0.38): pyramiding doubled drawdown on unseen coins — classic overfitting.
- **v2 edge confirmed on unseen coins** (expectancy ex-top-5 > 0.1R, profit factor 2.2, 5 of 7 trading years positive), but about **3× smaller** than on the development coins — survivorship bias in the original coin choice.
- **H2** is the most promising variant but was not the pre-registered candidate → tracked live as challenger instead of adopted.
- Validation criteria were tightened *before* seeing unseen-coin data after a test on random-walk coins showed that trend-following can show positive expectancy from a few extreme trades alone. The "positive years" metric initially counted years spent fully in cash (0%) as failures; this measurement bug was corrected **after** seeing the result and is disclosed here.

### Full 19-coin universe (reference for live trading)

| Version | Risk | CAGR | Max DD | Calmar | OOS Calmar |
|---|---|---|---|---|---|
| v2 | 0.5% | 16.4% | −17.7% | 0.93 | 1.18 |
| v2 | 1.0% | 25.9% | −30.1% | 0.86 | 0.91 |
| h2 | 0.5% | 23.5% | −19.9% | 1.18 | 1.07 |
| h2 | 1.0% | 37.6% | −34.9% | 1.08 | 0.87 |

Doubling risk raises returns but lowers risk-adjusted performance → **0.5% is the chosen risk level**.

## Limitations

- Survivorship bias: both coin sets are coins that exist today.
- The regime gate (006 v2) was itself chosen after seeing 2020–2026 data.
- Daily candles from one exchange; no funding, slippage beyond fees, or liquidity limits for small caps.
- Few independent bull cycles (2017, 2020–21, 2023–25).

## Usage

```bash
pip install -r requirements.txt
python fetch_data.py                                                          # development coins -> data/
python fetch_data.py --symbols LTC BCH TRX DOT ATOM ETC XLM NEAR FIL UNI --out data_val
python bull_system.py                  # v2 backtest + benchmarks + criteria   (--version v1 | h2, --risk 1)
python improve.py                      # round-2 development + unseen-coin validation
python signals.py                      # live paper-trading log (v2 and h2)
```

Requires project 006 next to this folder (`../006.BTC-REGIME-DETECTOR/resultados/estados_oficiales.csv`). On Windows, `run_daily.bat` refreshes data and writes the daily signals; schedule it after 006. `registro/senales_log.csv` is **append-only** and committed as a public forward record.

---
*Author: Ivan Alonso · [GitHub](https://github.com/ivanalonso-dev) · [LinkedIn](https://www.linkedin.com/in/ivanalonsodev) · [X](https://x.com/ivanalonsodev)*
