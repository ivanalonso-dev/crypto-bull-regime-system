#!/usr/bin/env python3
"""
crypto-bull-regime-system · ronda de mejora (pre-registrada)

LÍNEA BASE  v2 = solo sistema A (tendencia), sin sistema B.
HIPÓTESIS (fijadas antes de ver resultados)
  H1 piramidar      hasta 3 unidades; añade una cada +1 ATR desde la última compra,
                    cada unidad con el mismo riesgo; el stop del bloque sube.
  H2 salida holgada trailing de 4 x ATR (en vez de 3) y salida por mínimo de 20 días
                    (en vez de 10).
  H3 fuerza rel.    solo entra en las 3 monedas con mayor subida a 90 días.

PROCEDIMIENTO
  Paso 1 · desarrollo (9 monedas ya vistas, 2017-2026): cada hipótesis por separado.
           Pasa si mejora A LA VEZ el Calmar total y la esperanza por operación (R)
           frente a la línea base.
  Paso 2 · candidato = línea base + todas las hipótesis que pasen.
  Paso 3 · validación en 10 monedas NUNCA usadas (data_val/):
           · la v2 se CONFIRMA si: esperanza sin las 5 mejores operaciones > +0,1R,
             profit factor > 1,5 y mayoría de años positivos (de los años con operaciones).
           · el candidato se ADOPTA si: Calmar >= 1,2 x el de la v2 y esperanza sin las
             5 mejores > la de la v2 (y > 0).
           "Años positivos" cuenta solo los años en los que el sistema operó. La
           primera ejecución contaba también los años en liquidez (0%) como no
           positivos; fue un error de medición, corregido después de ver el resultado
           y documentado en el README.
           (Criterios endurecidos tras una prueba con monedas aleatorias, en la que el
           seguimiento de tendencia daba esperanza positiva solo por unas pocas
           operaciones extremas. Fijados antes de ver los datos reales de validación.)
  Las decisiones se toman con riesgo 0,5%; el 1% se muestra para valorar el tamaño.

Uso:
    python fetch_data.py --symbols LTC BCH TRX DOT ATOM ETC XLM NEAR FIL UNI --out data_val
    python improve.py
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

import bull_system as bs

V2 = bs.Params(use_b=False)
HYP = {
    "H1 piramidar": dict(pyr_max_units=3, pyr_step_atr=1.0),
    "H2 salida holgada": dict(a_trail_atr=4.0, dc_exit=20),
    "H3 fuerza relativa": dict(rs_top=3),
}
RISKS = [0.5, 1.0]
COLS = ["CAGR_%", "maxDD_%", "Calmar", "Sharpe"]


def evaluate(coins, states, p: bs.Params, start: str, label: str, set_name: str) -> dict:
    eq, tr, expo, *_ = bs.backtest(coins, states, p)
    eq = eq[eq.index >= start]
    m = bs.perf(eq)
    yearly = eq.resample("YE").last().pct_change()
    yearly.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    # solo cuentan los años en los que el sistema operó (en liquidez todo el año = 0%)
    active_years = set(pd.to_datetime(tr["entrada"]).dt.year) if len(tr) else set()
    yearly = yearly[[d.year in active_years for d in yearly.index]]
    return {"conjunto": set_name, "variante": label, "riesgo_%": p.risk_pct,
            **{k: round(m.get(k, np.nan), 2) for k in COLS},
            "trades": len(tr),
            "esperanza_R": round(tr["R"].mean(), 3) if len(tr) else np.nan,
            "PF": round(tr.loc[tr.R > 0, "R"].sum() / -tr.loc[tr.R <= 0, "R"].sum(), 2)
            if len(tr) and (tr.R <= 0).any() else np.nan,
            "esperanza_sin_top5_R": round(tr.drop(tr["R"].nlargest(5).index)["R"].mean(), 3)
            if len(tr) > 10 else np.nan,
            "años_positivos": f"{int((yearly > 0).sum())}/{len(yearly)}",
            "exposición_%": round(100 * expo[expo.index >= start].mean(), 1)}


def bench_rows(coins, states, btc, start, set_name) -> list[dict]:
    dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in coins.values()])))
    rows = []
    for name, eq in bs.benchmarks(coins, states, dates, V2.fee_pct, btc_df=btc).items():
        m = bs.perf(eq[eq.index >= start])
        rows.append({"conjunto": set_name, "variante": f"REF · {name}", "riesgo_%": np.nan,
                     **{k: round(m.get(k, np.nan), 2) for k in COLS}})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Ronda de mejora pre-registrada del 007")
    ap.add_argument("--data", default="data")
    ap.add_argument("--data-val", default="data_val")
    ap.add_argument("--states", default=bs.DEFAULT_STATES)
    ap.add_argument("--start", default="2017-08-17")
    ap.add_argument("--out", default="resultados")
    a = ap.parse_args()

    dev = bs.load_coins(a.data)
    val = bs.load_coins(a.data_val)
    if not val:
        raise SystemExit(f"No hay monedas en {a.data_val}/. Descárgalas con:\n  python fetch_data.py "
                         "--symbols LTC BCH TRX DOT ATOM ETC XLM NEAR FIL UNI --out data_val")
    overlap = set(dev) & set(val)
    if overlap:
        raise SystemExit(f"Las monedas de validación no pueden estar en desarrollo: {overlap}")
    states = bs.load_states(a.states)
    btc = dev["BTC"]
    print(f"Desarrollo: {', '.join(dev)}")
    print(f"Validación: {', '.join(f'{s} ({df.index[0].date()})' for s, df in val.items())}")

    rows = []
    # Paso 1 · desarrollo
    dev_res = {}
    for r in RISKS:
        base = evaluate(dev, states, replace(V2, risk_pct=r), a.start, "v2 (solo A)", "DESARROLLO")
        rows.append(base)
        dev_res[("v2", r)] = base
        for h, kw in HYP.items():
            res = evaluate(dev, states, replace(V2, risk_pct=r, **kw), a.start, h, "DESARROLLO")
            rows.append(res)
            dev_res[(h, r)] = res
    passing = [h for h in HYP
               if dev_res[(h, 0.5)]["Calmar"] > dev_res[("v2", 0.5)]["Calmar"]
               and dev_res[(h, 0.5)]["esperanza_R"] > dev_res[("v2", 0.5)]["esperanza_R"]]

    # Paso 2 · candidato
    cand_kw = {}
    for h in passing:
        cand_kw.update(HYP[h])
    cand_label = "CANDIDATO (v2 + " + " + ".join(h.split()[0] for h in passing) + ")" if passing else None
    if passing:
        for r in RISKS:
            rows.append(evaluate(dev, states, replace(V2, risk_pct=r, **cand_kw), a.start, cand_label, "DESARROLLO"))
    rows += bench_rows(dev, states, btc, a.start, "DESARROLLO")

    # Paso 3 · validación en monedas nuevas
    val_res = {}
    for r in RISKS:
        v = evaluate(val, states, replace(V2, risk_pct=r), a.start, "v2 (solo A)", "VALIDACIÓN")
        rows.append(v)
        val_res[("v2", r)] = v
        for h, kw in HYP.items():
            rows.append(evaluate(val, states, replace(V2, risk_pct=r, **kw), a.start, h, "VALIDACIÓN"))
        if passing:
            c = evaluate(val, states, replace(V2, risk_pct=r, **cand_kw), a.start, cand_label, "VALIDACIÓN")
            rows.append(c)
            val_res[("cand", r)] = c
    rows += bench_rows(val, states, btc, a.start, "VALIDACIÓN")

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "mejora.csv"), index=False)

    show = ["variante", "riesgo_%", "CAGR_%", "maxDD_%", "Calmar", "Sharpe", "trades", "esperanza_R",
            "esperanza_sin_top5_R", "PF", "años_positivos", "exposición_%"]
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 40):
        for conj in ["DESARROLLO", "VALIDACIÓN"]:
            print(f"\n=== {conj} ===")
            print(df[df.conjunto == conj][show].to_string(index=False))

    print("\n=== DECISIONES (riesgo 0,5%) ===")
    b = dev_res[("v2", 0.5)]
    for h in HYP:
        d = dev_res[(h, 0.5)]
        ok = h in passing
        print(f"  [{'PASA' if ok else 'NO  '}] {h}: Calmar {d['Calmar']} vs {b['Calmar']}, "
              f"esperanza {d['esperanza_R']}R vs {b['esperanza_R']}R")
    v2v = val_res[("v2", 0.5)]
    yrs_pos = int(str(v2v["años_positivos"]).split("/")[0])
    yrs_tot = int(str(v2v["años_positivos"]).split("/")[1])
    confirm = v2v["esperanza_sin_top5_R"] > 0.1 and v2v["PF"] > 1.5 and yrs_pos > yrs_tot / 2
    print(f"\n  v2 en monedas nuevas: esperanza {v2v['esperanza_R']}R (sin top5 {v2v['esperanza_sin_top5_R']}R), "
          f"PF {v2v['PF']}, años positivos {v2v['años_positivos']}, Calmar {v2v['Calmar']}")
    print(f"    -> {'la ventaja del sistema A SE CONFIRMA en monedas nuevas' if confirm else 'la ventaja del sistema A NO se confirma en monedas nuevas'}")
    if passing:
        cv = val_res[("cand", 0.5)]
        adopt = (cv["Calmar"] >= 1.2 * v2v["Calmar"] and cv["esperanza_sin_top5_R"] > 0
                 and cv["esperanza_sin_top5_R"] > v2v["esperanza_sin_top5_R"])
        print(f"  {cand_label} en monedas nuevas: Calmar {cv['Calmar']} vs {v2v['Calmar']}, "
              f"esperanza sin top5 {cv['esperanza_sin_top5_R']}R vs {v2v['esperanza_sin_top5_R']}R")
        print(f"    -> {'SE ADOPTA' if adopt else 'NO se adopta; se mantiene la v2'}")
    else:
        print("  Ninguna hipótesis mejora en desarrollo -> se mantiene la v2.")
    print(f"\nTabla completa en ./{a.out}/mejora.csv")


if __name__ == "__main__":
    main()
