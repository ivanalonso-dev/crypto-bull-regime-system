#!/usr/bin/env python3
"""
crypto-bull-regime-system · registro diario de señales (paper trading)

Ejecuta la versión oficial (v2) y la aspirante (h2) sobre el universo completo
(data/ + data_val/, 19 monedas) con los datos más recientes y el estado del 006,
y anota cada día, sin reescribir nunca el pasado:
  - el estado del 006
  - el capital simulado de cada versión
  - las posiciones abiertas (entrada, stop actual)
  - las órdenes para la próxima apertura (compras, ventas y motivo)

registro/senales_log.csv      una fila por fecha y versión (append-only)
registro/posiciones_<v>.csv   foto actual de las posiciones de cada versión

Es la validación en vivo del sistema: dentro de unos meses se compara v2 frente a
h2 y frente a tener BTC en régimen alcista, con datos que no existían al diseñarlo.

Uso:
    python signals.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import bull_system as bs

STATE_NAMES = {2: "ALCISTA FUERTE", 1: "ALCISTA DÉBIL / LATERAL", 0: "NO ALCISTA"}
LIVE_VERSIONS = ["v2", "h2"]


def snapshot(coins, states, version: str) -> tuple[dict, pd.DataFrame]:
    out: dict = {}
    eq, tr, *_ = bs.backtest(coins, states, bs.VERSIONS[version], state_out=out)
    syms, close = out["syms"], out["close"]
    rows = []
    for j, q in out["pos"].items():
        rows.append({"moneda": syms[j], "sistema": q["sys"], "entrada": eq.index[q["t0"]].date(), "precio_entrada": round(q["entry"], 6),
                     "stop": round(q["stop"], 6), "cierre": round(float(close[j]), 6),
                     "R_abierto": round((close[j] * q["qty"] - q["cost"]) / q["risk_usd"], 2),
                     "salida_mañana": q["exit"] or ""})
    positions = pd.DataFrame(rows)
    orders = [f"COMPRA {syms[j]}" for j, kind, *_ in out["pending"] if kind == "new"]
    orders += [f"VENTA {syms[j]} ({q['exit']})" for j, q in out["pos"].items() if q["exit"]]
    row = {"fecha": str(out["date"].date()), "version": version,
           "estado_006": STATE_NAMES.get(int(out["state"]), "n/d"),
           "capital_simulado": round(float(out["equity"]), 2),
           "rentabilidad_total_%": round(100 * (out["equity"] / bs.Params().capital - 1), 2),
           "n_posiciones": len(rows),
           "posiciones": "; ".join(f"{r['moneda']} stop {r['stop']:g}" for r in rows),
           "ordenes_apertura": "; ".join(orders),
           "registrado_en": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    return row, positions


def main():
    ap = argparse.ArgumentParser(description="Registro diario de señales del 007")
    ap.add_argument("--data", nargs="+", default=["data", "data_val"])
    ap.add_argument("--states", default=bs.DEFAULT_STATES)
    ap.add_argument("--log-dir", default="registro")
    a = ap.parse_args()

    coins = {}
    for folder in a.data:
        coins.update(bs.load_coins(folder))
    states = bs.load_states(a.states)
    os.makedirs(a.log_dir, exist_ok=True)
    log_path = os.path.join(a.log_dir, "senales_log.csv")
    log = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()

    for v in LIVE_VERSIONS:
        row, positions = snapshot(coins, states, v)
        positions.to_csv(os.path.join(a.log_dir, f"posiciones_{v}.csv"), index=False)
        done = len(log) and ((log["fecha"].astype(str) == row["fecha"]) & (log["version"] == v)).any()
        if not done:
            log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        print(f"\n[{v}] {row['fecha']} · 006: {row['estado_006']} · capital simulado "
              f"{row['capital_simulado']:,.0f} ({row['rentabilidad_total_%']:+.1f}% desde 2017)")
        print(f"  Posiciones ({row['n_posiciones']}): {row['posiciones'] or '—'}")
        print(f"  Órdenes para la próxima apertura: {row['ordenes_apertura'] or '—'}")
        if done:
            print("  (fecha ya registrada: no se modifica el registro)")
    log.to_csv(log_path, index=False)
    print(f"\nRegistro: {log_path} ({len(log)} filas)")


if __name__ == "__main__":
    main()
