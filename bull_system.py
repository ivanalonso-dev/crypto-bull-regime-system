#!/usr/bin/env python3
"""
crypto-bull-regime-system · backtester de cartera

Opera una cesta de criptomonedas SOLO cuando el detector del proyecto 006 dice
que BTC está en régimen alcista, con un sistema distinto según el sub-estado:

  ALCISTA FUERTE           -> Sistema A (tendencia, ruptura de Donchian)
  ALCISTA DÉBIL / LATERAL  -> Sistema B (reversión a la media, RSI de 2 periodos)
  NO ALCISTA               -> liquidez (lo cubrirá el proyecto 008)

REGLAS (pre-registradas)
------------------------
Sistema A  entrada: cierre > máximo de los cierres de los N días previos (N = 20)
           stop inicial 3 x ATR(20); trailing (chandelier): máximo cierre desde la
           entrada - 3 x ATR; salida si el cierre < mínimo de los 10 días previos.
           Solo abre en ALCISTA FUERTE; las posiciones abiertas siguen su gestión
           si el estado pasa a DÉBIL.
Sistema B  entrada: RSI(2) < 10. Stop 2,5 x ATR(20). Salida si el cierre > media de
           5 días o a los 10 días. Solo abre en ALCISTA DÉBIL / LATERAL.
Ambos      cierran todo si el estado pasa a NO ALCISTA.

Gestión    riesgo 0,5% del capital por operación (tamaño = riesgo / distancia al
           stop), máx. 20% del capital por moneda y 100% en total, sin apalancamiento.
           Señal al cierre, ejecución en la apertura siguiente; los stops son órdenes
           en mercado (se ejecutan intradía, con hueco si abre por debajo).
           Comisión 0,10% por lado.

VALIDACIÓN
----------
In-sample < --split (2023-01-01) <= out-of-sample. Referencias: comprar y mantener
BTC, el 006 solo (BTC en estado alcista) y una cesta equiponderada de las mismas
monedas mantenida en estado alcista. Criterios de éxito comprobados al final.

VERSIONES: v1 (A + B) no pasó la validación (3/5 criterios: el sistema B no tiene
ventaja). v2 (solo A) es la versión oficial. h2 = v2 con salida holgada, en
seguimiento en vivo como aspirante (ver improve.py y signals.py).

Uso:
    python bull_system.py                  # v2 oficial
    python bull_system.py --version v1     # reproduce la v1 pre-registrada
    python bull_system.py --states ../006.BTC-REGIME-DETECTOR/resultados/estados_oficiales.csv
    python bull_system.py --synthetic       # prueba sin datos
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

YEAR = 365
DEFAULT_STATES = os.path.join("..", "006.BTC-REGIME-DETECTOR", "resultados", "estados_oficiales.csv")
ORDER = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "LINK", "AVAX"]


@dataclass
class Params:
    dc_entry: int = 20
    dc_exit: int = 10
    atr_n: int = 20
    a_stop_atr: float = 3.0
    b_rsi_thr: float = 10.0
    b_stop_atr: float = 2.5
    b_max_days: int = 10
    b_exit_sma: int = 5
    risk_pct: float = 0.5
    max_pos_pct: float = 20.0
    max_total_pct: float = 100.0
    fee_pct: float = 0.10
    warmup: int = 60
    capital: float = 100_000.0
    use_b: bool = True             # v1 = A + B; v2 = solo A
    a_trail_atr: float = 3.0       # trailing (chandelier) del sistema A
    pyr_max_units: int = 1         # 1 = sin piramidar
    pyr_step_atr: float = 1.0      # añade unidad cada +N ATR desde la última compra
    rs_top: int = 0                # 0 = sin filtro; k = solo las k monedas con mayor subida
    rs_lookback: int = 90


VERSIONS = {
    "v1": Params(),                                            # A + B (no validada)
    "v2": Params(use_b=False),                                 # OFICIAL: solo A
    "h2": Params(use_b=False, a_trail_atr=4.0, dc_exit=20),    # aspirante: salida holgada
}


# -----------------------------------------------------------------------------
# Datos
# -----------------------------------------------------------------------------
def load_coins(folder: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if symbols and sym not in symbols:
            continue
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        out[sym] = df[["open", "high", "low", "close"]].astype(float)
    order = [s for s in ORDER if s in out] + sorted(s for s in out if s not in ORDER)
    return {s: out[s] for s in order}


def load_states(path: str) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = s.index.normalize()
    return s


def synthetic(seed: int = 11) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Paseos aleatorios correlacionados (sin ventaja posible) + estados derivados de 'BTC'."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-08-17", "2026-10-03", freq="D")
    n = len(idx)
    mkt = rng.standard_t(4, n) * 0.025
    coins = {}
    for k, sym in enumerate(ORDER):
        start = 0 if k < 5 else 900 + 60 * k
        r = 0.7 * mkt + rng.standard_t(4, n) * 0.02
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.015, n)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.015, n)))
        coins[sym] = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close},
                                  index=idx).iloc[start:]
    btc = coins["BTC"]["close"]
    trend = (btc > btc.rolling(200).mean()).astype(float)
    er = (btc - btc.shift(90)).abs() / btc.diff().abs().rolling(90).sum()
    states = pd.Series(np.where(trend == 1, 1 + (er > 0.14), 0), index=idx, dtype=float)
    states[btc.rolling(200).mean().isna()] = np.nan
    return coins, states


# -----------------------------------------------------------------------------
# Indicadores
# -----------------------------------------------------------------------------
def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def panel(coins: dict[str, pd.DataFrame], p: Params) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in coins.values()])))
    fields = {k: [] for k in ["o", "h", "l", "c", "atr", "dc_hi", "dc_lo", "rsi", "sma", "ok", "mom"]}
    for df in coins.values():
        df = df.reindex(dates)
        c, h, l = df["close"], df["high"], df["low"]
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        age = c.notna().cumsum()
        fields["o"].append(df["open"]); fields["h"].append(h); fields["l"].append(l); fields["c"].append(c)
        fields["atr"].append(tr.rolling(p.atr_n).mean())
        fields["dc_hi"].append(c.rolling(p.dc_entry).max().shift(1))
        fields["dc_lo"].append(c.rolling(p.dc_exit).min().shift(1))
        fields["rsi"].append(rsi(c, 2))
        fields["sma"].append(c.rolling(p.b_exit_sma).mean())
        fields["ok"].append((age >= p.warmup) & c.notna())
        fields["mom"].append(c / c.shift(p.rs_lookback) - 1)
    return dates, {k: pd.concat(v, axis=1).values for k, v in fields.items()}


# -----------------------------------------------------------------------------
# Motor
# -----------------------------------------------------------------------------
def backtest(coins: dict[str, pd.DataFrame], states: pd.Series, p: Params,
             state_out: dict | None = None):
    syms = list(coins)
    dates, X = panel(coins, p)
    st = states.reindex(dates).ffill(limit=3).fillna(0).values
    T, N = X["c"].shape
    fee = p.fee_pct / 100
    cash = p.capital
    pos: dict[int, dict] = {}
    pending: list[tuple] = []       # (j, tipo, sistema, dist, qty)  tipo = "new" | "add"
    trades, equity, exposure = [], np.full(T, np.nan), np.zeros(T)

    def close_pos(j, px, t, reason):
        nonlocal cash
        q = pos.pop(j)
        proceeds = q["qty"] * px * (1 - fee)
        cash += proceeds
        pnl = proceeds - q["cost"]
        trades.append({"moneda": syms[j], "sistema": q["sys"], "entrada": dates[q["t0"]].date(),
                       "salida": dates[t].date(), "precio_entrada": q["entry"], "precio_salida": px,
                       "dias": t - q["t0"], "unidades": q["units"], "R": pnl / q["risk_usd"],
                       "pnl": pnl, "pnl_%_capital": 100 * pnl / q["eq0"], "motivo": reason})

    for t in range(T):
        o, h, l, c = X["o"][t], X["h"][t], X["l"][t], X["c"][t]
        # 1) apertura: salidas pendientes, entradas y ampliaciones decididas ayer
        for j in [j for j, q in pos.items() if q.get("exit")]:
            if not np.isnan(o[j]):
                close_pos(j, o[j], t, pos[j]["exit"])
        for j, kind, sys, dist, qty in pending:
            if np.isnan(o[j]) or qty <= 0:
                continue
            if kind == "new" and j in pos:
                continue
            if kind == "add" and (j not in pos or pos[j].get("exit")):
                continue
            qty = min(qty, cash / (o[j] * (1 + fee)))
            if qty * o[j] < 0.005 * p.capital:
                continue
            cost = qty * o[j] * (1 + fee)
            cash -= cost
            if kind == "new":
                eq_now = cash + cost + sum(q["qty"] * o[k] for k, q in pos.items() if not np.isnan(o[k]))
                pos[j] = {"sys": sys, "qty": qty, "entry": o[j], "t0": t, "risk_usd": qty * dist,
                          "stop": o[j] - dist, "hi": o[j], "exit": None, "cost": cost,
                          "units": 1, "last_add": o[j], "eq0": eq_now}
            else:
                q = pos[j]
                q["entry"] = (q["entry"] * q["qty"] + o[j] * qty) / (q["qty"] + qty)
                q["qty"] += qty
                q["cost"] += cost
                q["units"] += 1
                q["last_add"] = o[j]
                q["stop"] = max(q["stop"], o[j] - dist)
        pending = []
        # 2) intradía: stops
        for j in list(pos):
            q = pos[j]
            if not np.isnan(l[j]) and l[j] <= q["stop"]:
                close_pos(j, min(o[j], q["stop"]) if not np.isnan(o[j]) else q["stop"], t, "STOP")
        # 3) cierre: valoración, gestión y nuevas señales
        val = sum(q["qty"] * (c[j] if not np.isnan(c[j]) else q["entry"]) for j, q in pos.items())
        eq = cash + val
        equity[t], exposure[t] = eq, val / eq if eq > 0 else 0
        s = st[t]
        for j, q in pos.items():
            if np.isnan(c[j]):
                continue
            q["hi"] = max(q["hi"], c[j])
            days = t - q["t0"] + 1
            reason = None
            if s == 0:
                reason = "REGIMEN"
            elif q["sys"] == "A":
                q["stop"] = max(q["stop"], q["hi"] - p.a_trail_atr * X["atr"][t, j])
                if c[j] < X["dc_lo"][t, j]:
                    reason = "SALIDA_DONCHIAN"
            else:
                if c[j] > X["sma"][t, j]:
                    reason = "SALIDA_MEDIA"
                elif days >= p.b_max_days:
                    reason = "SALIDA_TIEMPO"
            q["exit"] = reason
        if s >= 1 and t < T - 1:
            live = sum(q["qty"] * c[j] for j, q in pos.items() if not q["exit"] and not np.isnan(c[j]))
            room = p.max_total_pct / 100 * eq - live
            # ampliaciones (piramidación) del sistema A
            if p.pyr_max_units > 1:
                for j, q in pos.items():
                    if (q["sys"] != "A" or q["exit"] or q["units"] >= p.pyr_max_units
                            or np.isnan(c[j]) or np.isnan(X["atr"][t, j])):
                        continue
                    if c[j] >= q["last_add"] + p.pyr_step_atr * X["atr"][t, j]:
                        dist = p.a_stop_atr * X["atr"][t, j]
                        qty = (p.risk_pct / 100 * eq) / dist
                        cap_coin = p.max_pos_pct / 100 * eq - q["qty"] * c[j]
                        notional = min(qty * c[j], cap_coin, room)
                        if notional >= 0.01 * eq:
                            pending.append((j, "add", "A", dist, notional / c[j]))
                            room -= notional
            # filtro de fuerza relativa
            allowed = np.ones(N, dtype=bool)
            if p.rs_top > 0:
                mom = np.where(X["ok"][t] & ~np.isnan(X["mom"][t]), X["mom"][t], -np.inf)
                ranks = np.argsort(-mom)
                allowed[:] = False
                allowed[ranks[:p.rs_top]] = True
                allowed &= np.isfinite(mom)
            for j in range(N):
                if j in pos or not X["ok"][t, j] or np.isnan(X["atr"][t, j]) or room <= 0.01 * eq:
                    continue
                sig = None
                if s == 2 and c[j] > X["dc_hi"][t, j] and allowed[j]:
                    sig, k = "A", p.a_stop_atr
                elif s == 1 and p.use_b and X["rsi"][t, j] < p.b_rsi_thr:
                    sig, k = "B", p.b_stop_atr
                if sig is None:
                    continue
                dist = k * X["atr"][t, j]
                qty = (p.risk_pct / 100 * eq) / dist
                notional = min(qty * c[j], p.max_pos_pct / 100 * eq, room)
                if notional < 0.01 * eq:
                    continue
                pending.append((j, "new", sig, dist, notional / c[j]))
                room -= notional

    if state_out is not None:
        state_out.update({"syms": syms, "date": dates[-1], "pos": pos, "pending": pending,
                          "close": X["c"][-1], "atr": X["atr"][-1], "state": st[-1],
                          "equity": equity[-1], "cash": cash})
    eq = pd.Series(equity, index=dates).ffill()
    return eq, pd.DataFrame(trades), pd.Series(exposure, index=dates), dates, st


# -----------------------------------------------------------------------------
# Referencias y métricas
# -----------------------------------------------------------------------------
def benchmarks(coins: dict[str, pd.DataFrame], states: pd.Series, dates, fee_pct: float,
               btc_df: pd.DataFrame | None = None) -> dict:
    fee = fee_pct / 100
    st = states.reindex(dates).ffill(limit=3).fillna(0)
    on = (st >= 1).astype(float).shift(1).fillna(0)
    switch = on.diff().abs().fillna(on)
    btc = (btc_df if btc_df is not None else coins["BTC"])["close"].reindex(dates)
    r_btc = btc.pct_change().fillna(0)
    rets = pd.concat([df["close"].reindex(dates).pct_change() for df in coins.values()], axis=1)
    basket = rets.mean(axis=1, skipna=True).fillna(0)
    to_eq = lambda r: (1 + r).cumprod()
    return {
        "Comprar y mantener BTC": to_eq(r_btc),
        "006 solo (BTC en alcista)": to_eq(on * r_btc - switch * fee),
        "Cesta en alcista (equiponderada)": to_eq(on * basket - switch * fee),
    }


def perf(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 30:
        return {}
    r = eq.pct_change().fillna(0)
    years = len(eq) / YEAR
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(YEAR)
    return {"CAGR_%": 100 * cagr, "maxDD_%": 100 * dd,
            "Sharpe": r.mean() * YEAR / vol if vol > 0 else np.nan,
            "Calmar": cagr / abs(dd) if dd < 0 else np.nan}


def segments(eq: pd.Series, start: str, split: str) -> dict:
    out = {}
    for tag, lo, hi in [("TOTAL", start, None), ("IS", start, split), ("OOS", split, None)]:
        e = eq[(eq.index >= lo) & ((eq.index < hi) if hi else True)]
        for k, v in perf(e).items():
            out[f"{tag} {k}"] = v
    return out


def trade_stats(tr: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame()
    g = tr.groupby(by)["R"]
    out = pd.DataFrame({
        "trades": g.size(),
        "win_%": g.apply(lambda r: 100 * (r > 0).mean()),
        "R_medio_ganador": g.apply(lambda r: r[r > 0].mean()),
        "R_medio_perdedor": g.apply(lambda r: r[r <= 0].mean()),
        "esperanza_R": g.mean(),
        "total_R": g.sum(),
        "PF": g.apply(lambda r: r[r > 0].sum() / -r[r <= 0].sum() if (r <= 0).any() else np.nan),
    })
    return out.round(3)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_all(coins, states, p: Params, start: str, split: str, out: str, grid: bool = True):
    os.makedirs(out, exist_ok=True)
    eq, tr, expo, dates, st = backtest(coins, states, p)
    first = max(pd.Timestamp(start), dates[0])
    start = str(first.date())
    rows = {"SISTEMA 007": segments(eq, start, split)}
    bench = benchmarks(coins, states, dates, p.fee_pct)
    for name, b in bench.items():
        rows[name] = segments(b, start, split)
    metrics = pd.DataFrame(rows).T
    tr["tramo"] = np.where(pd.to_datetime(tr["entrada"]) < pd.Timestamp(split), "IS", "OOS") if len(tr) else []
    sys_stats = trade_stats(tr, ["sistema", "tramo"]) if len(tr) else pd.DataFrame()
    sys_tot = trade_stats(tr, ["sistema"]) if len(tr) else pd.DataFrame()
    coin_stats = trade_stats(tr, ["moneda"]) if len(tr) else pd.DataFrame()

    single = {}
    for sym in ["BTC", "ETH"]:
        if sym in coins:
            e1, t1, *_ = backtest({sym: coins[sym]}, states, p)
            single[sym] = segments(e1, start, split)
    single_df = pd.DataFrame(single).T

    rob = []
    if grid:
        for dc in [15, 20, 30]:
            for thr in [5.0, 10.0, 15.0]:
                e2, *_ = backtest(coins, states, replace(p, dc_entry=dc, b_rsi_thr=thr))
                sg = segments(e2, start, split)
                rob.append({"donchian": dc, "rsi_umbral": thr, **{k: sg.get(k) for k in
                            ["IS CAGR_%", "IS Calmar", "OOS CAGR_%", "OOS maxDD_%", "OOS Calmar"]}})
    rob_df = pd.DataFrame(rob)

    # criterios pre-registrados
    m = metrics
    oos_cal = m.loc["SISTEMA 007", "OOS Calmar"]
    crit = [
        ("1. OOS Calmar > las 3 referencias",
         all(oos_cal > m.loc[b, "OOS Calmar"] for b in bench), f"{oos_cal:.2f} vs " +
         ", ".join(f"{m.loc[b, 'OOS Calmar']:.2f}" for b in bench)),
        ("2. Drawdown máximo total < 30%", m.loc["SISTEMA 007", "TOTAL maxDD_%"] > -30,
         f"{m.loc['SISTEMA 007', 'TOTAL maxDD_%']:.1f}%"),
        ("3. Positivo en BTC y en ETH por separado (OOS)",
         all(single_df.loc[s, "OOS CAGR_%"] > 0 for s in single_df.index) if len(single_df) else False,
         ", ".join(f"{s} {single_df.loc[s, 'OOS CAGR_%']:.1f}%" for s in single_df.index)),
        ("4. Robustez: OOS positivo en los 9 vecinos",
         bool(len(rob_df)) and bool((rob_df["OOS CAGR_%"] > 0).all()),
         f"mín OOS CAGR {rob_df['OOS CAGR_%'].min():.1f}%" if len(rob_df) else "sin rejilla"),
        ("5. Esperanza positiva en cada subsistema",
         bool(len(sys_tot)) and bool((sys_tot["esperanza_R"] > 0).all()),
         ", ".join(f"{k} {v:+.3f}R" for k, v in sys_tot["esperanza_R"].items()) if len(sys_tot) else "sin trades"),
    ]
    crit_df = pd.DataFrame(crit, columns=["criterio", "cumple", "detalle"])

    # guardar
    metrics.round(3).to_csv(os.path.join(out, "metricas.csv"))
    tr.to_csv(os.path.join(out, "trades.csv"), index=False)
    sys_stats.to_csv(os.path.join(out, "sistemas.csv"))
    coin_stats.to_csv(os.path.join(out, "monedas.csv"))
    single_df.round(3).to_csv(os.path.join(out, "btc_eth_por_separado.csv"))
    rob_df.round(3).to_csv(os.path.join(out, "robustez.csv"), index=False)
    crit_df.to_csv(os.path.join(out, "criterios.csv"), index=False)
    curves = pd.DataFrame({"SISTEMA 007": eq / eq.iloc[0], **bench, "exposicion": expo, "estado": st},
                          index=dates)
    curves[curves.index >= first].to_csv(os.path.join(out, "curvas.csv"))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(p), "start": start, "split": split, "monedas": list(coins)}, f, indent=2)
    return metrics, sys_stats, sys_tot, coin_stats, single_df, rob_df, crit_df, expo


def main():
    ap = argparse.ArgumentParser(description="Sistema para régimen alcista (proyecto 007)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--states", default=DEFAULT_STATES, help="estados_oficiales.csv del proyecto 006")
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--split", default="2023-01-01")
    ap.add_argument("--no-grid", action="store_true", help="omite la rejilla de robustez")
    ap.add_argument("--version", default="v2", choices=list(VERSIONS),
                    help="v1 = A+B (pre-registrada, no validada), v2 = solo A (oficial), h2 = aspirante")
    ap.add_argument("--risk", type=float, help="riesgo por operación en %% (por defecto 0,5)")
    ap.add_argument("--out", default="resultados")
    a = ap.parse_args()

    if a.synthetic:
        coins, states = synthetic()
    else:
        coins = load_coins(a.data, a.symbols)
        if not coins:
            raise SystemExit(f"No hay CSV en {a.data}/. Ejecuta primero: python fetch_data.py")
        if not os.path.exists(a.states):
            raise SystemExit(f"No encuentro {a.states}. Ejecuta el proyecto 006 o indica --states.")
        states = load_states(a.states)
    print(f"Monedas: {', '.join(f'{s} ({df.index[0].date()})' for s, df in coins.items())}")

    p = VERSIONS[a.version]
    if a.risk:
        p = replace(p, risk_pct=a.risk)
    print(f"Versión: {a.version}  ·  riesgo {p.risk_pct}% por operación")
    metrics, sys_stats, sys_tot, coin_stats, single_df, rob_df, crit_df, expo = run_all(
        coins, states, p, a.start, a.split, a.out, grid=not a.no_grid)

    show = ["TOTAL CAGR_%", "TOTAL maxDD_%", "TOTAL Calmar", "IS CAGR_%", "IS maxDD_%", "IS Calmar",
            "OOS CAGR_%", "OOS maxDD_%", "OOS Calmar", "OOS Sharpe"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:,.2f}".format):
        print(f"\n=== CARTERA vs REFERENCIAS (IS < {a.split} <= OOS) ===")
        print(metrics[[c for c in show if c in metrics]].to_string())
        print(f"  Exposición media: {100 * expo.mean():.1f}% del capital")
        print("\n=== SUBSISTEMAS (en R) ===")
        print(sys_stats.to_string() if len(sys_stats) else "  sin trades")
        print("\n=== POR MONEDA ===")
        print(coin_stats.to_string() if len(coin_stats) else "  sin trades")
        print("\n=== BTC y ETH POR SEPARADO ===")
        print(single_df[[c for c in show if c in single_df]].to_string())
        if len(rob_df):
            print("\n=== ROBUSTEZ (Donchian x umbral RSI) ===")
            print(rob_df.to_string(index=False))
    print("\n=== CRITERIOS PRE-REGISTRADOS ===")
    for _, r in crit_df.iterrows():
        print(f"  [{'CUMPLE' if r['cumple'] else 'NO    '}] {r['criterio']}: {r['detalle']}")
    n_ok = int(crit_df["cumple"].sum())
    print(f"\n  Resultado: {n_ok}/5 criterios. " +
          ("SISTEMA VALIDADO." if n_ok == 5 else "No pasa la validación completa."))
    print(f"\nFicheros en ./{a.out}/")


if __name__ == "__main__":
    main()
