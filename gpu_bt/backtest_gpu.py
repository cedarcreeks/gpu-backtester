"""
backtest_gpu.py — Backtest PIT acelerado por GPU (CuPy)
===============================================================================
Variante de the CPU engine but uses CuPy kernels for:
  - precompute candidatos (RS, indicadores) vectorizado en GPU
  - simulación OCO vectorizada en GPU sobre [N_signals, max_hold+1]

filtrar_candidatos + simular_portfolio se reusan del CPU (puro dict ops, ya
rápidos) para garantizar paridad de filtros/sizing/heat-cap.

Uso (nightly fast):
  python scripts/backtest_gpu.py --pit --desde 2024-01-01 --hasta 2025-12-31
  python scripts/backtest_gpu.py --pit --ventanas 4 --guardar
  python scripts/backtest_gpu.py --pit --estado BREAKOUT --guardar

Self-contained backtest entry. Drop-in for nightly analysis.
===============================================================================
"""
from __future__ import annotations

import sys as _sys, os as _os
if _sys.stdout and hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import datetime
import json
import pathlib
import time

from gpu_bt.core import (
    Params, filtrar_candidatos, simular_portfolio,
)
from gpu_bt.gpu import (
    gpu_disponible, historico_a_tensor, extract_candidatos_gpu,
    simulate_trades_gpu,
)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = _PROJECT_ROOT / "reports"


def parse_date(s: str) -> datetime.date:
    return datetime.date.fromisoformat(s)


def cargar_pit(desde: datetime.date, hasta: datetime.date) -> dict:
    try:
        from gpu_bt.pit_universe import PitUniverse
    except ImportError:
        print("ERROR: pit_universe.py no encontrado.")
        _sys.exit(1)
    pit = PitUniverse()
    pit.build(str(desde), str(hasta))
    historico = pit.historico
    if not historico:
        print("ERROR: PIT no pudo construir histórico.")
        _sys.exit(1)
    return historico


def run_gpu_backtest(historico: dict,
                     desde: datetime.date,
                     hasta: datetime.date,
                     params: Params,
                     estado_filtro: str = None) -> dict:
    t0 = time.time()
    print(f"[GPU] stack tensor histórico ({len(historico)} tickers)...")
    tensor = historico_a_tensor(historico)
    t1 = time.time()
    arr = tensor["arr_gpu"]
    N, T, _ = arr.shape
    print(f"[GPU] tensor [{N},{T},5] listo en {t1-t0:.1f}s")

    print(f"[GPU] extract candidatos PIT [{desde}..{hasta}]...")
    candidatos = extract_candidatos_gpu(tensor, desde, hasta)
    t2 = time.time()
    print(f"[GPU] {len(candidatos)} candidatos en {t2-t1:.1f}s")

    señales = filtrar_candidatos(candidatos, params, estado_filtro)
    t3 = time.time()
    print(f"[GPU] {len(señales)} señales tras filtros ({t3-t2:.1f}s)")
    if not señales:
        return {"error": "Sin señales", "n_señales": 0,
                "params": params.as_dict(), "n_operaciones": 0}

    operaciones = simulate_trades_gpu(señales, tensor, params)
    t4 = time.time()
    print(f"[GPU] sim {len(operaciones)} trades en {t4-t3:.1f}s")

    pf = simular_portfolio(operaciones, params)
    t5 = time.time()
    print(f"[GPU] portfolio sim en {t5-t4:.1f}s | TOTAL {t5-t0:.1f}s")

    validas = [o for o in operaciones if o["outcome"] != "SIN_DATOS"]
    wins = [o for o in validas if "WIN" in o["outcome"]]
    losses = [o for o in validas if o["outcome"] == "LOSS"]
    nv = len(wins) + len(losses)
    pnls = [o["pnl_pct"] for o in validas]
    import statistics
    avg_pnl = statistics.mean(pnls) if pnls else 0
    pw = [o["pnl_pct"] for o in wins]; pl = [o["pnl_pct"] for o in losses]
    wr_ops = len(wins) / nv * 100 if nv else 0
    pf_ops = sum(pw) / abs(sum(pl)) if pl and sum(pl) != 0 else 0

    return {
        "params":           params.as_dict(),
        "params_label":     params.label(),
        "n_señales":        len(señales),
        "n_operaciones":    len(validas),
        "wr_ops_pct":       round(wr_ops, 2),
        "pf_ops":           round(pf_ops, 3),
        "avg_pnl_pct":      round(avg_pnl, 3),
        "portfolio":        pf,
        "operaciones":      operaciones,
        "tiempos": {
            "stack_tensor": round(t1 - t0, 2),
            "precompute":   round(t2 - t1, 2),
            "filtrar":      round(t3 - t2, 2),
            "sim_trades":   round(t4 - t3, 2),
            "portfolio":    round(t5 - t4, 2),
            "total":        round(t5 - t0, 2),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pit",   action="store_true", default=True,
                    help="Universo PIT (default ON, único modo soportado).")
    ap.add_argument("--desde", type=parse_date,
                    default=datetime.date(2022, 1, 1))
    ap.add_argument("--hasta", type=parse_date,
                    default=datetime.date.today())
    ap.add_argument("--estado", type=str, default=None,
                    help="Filtrar BREAKOUT/SQUEEZE/EN BASE/EXTENDIDO/VIGILANCIA")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--rs-min", type=int, default=50)
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--guardar", action="store_true")
    args = ap.parse_args()

    if not gpu_disponible():
        print("ERROR: CuPy/GPU no disponible. Usa scripts/backtest_wf.py (CPU).")
        _sys.exit(1)

    params = Params(
        rs_minimo=args.rs_min,
        score_minimo=args.score_min,
        capital_total=args.capital,
    )

    historico = cargar_pit(args.desde, args.hasta)
    res = run_gpu_backtest(historico, args.desde, args.hasta,
                            params, args.estado)

    if "error" in res:
        print(f"[GPU] {res['error']}")
        return

    print()
    print("═" * 70)
    print(f"  Resultado GPU backtest PIT  [{args.desde}..{args.hasta}]")
    print("═" * 70)
    pf = res["portfolio"]
    print(f"  Capital inicial:    ${pf['capital_inicial']:,.0f}")
    print(f"  Capital final:      ${pf['capital_final']:,.2f}")
    print(f"  Retorno:            {pf['retorno_pct']:+.2f}%")
    print(f"  Max DD:             {pf['max_drawdown_pct']:.2f}% ({pf['max_dd_duration_d']}d)")
    print(f"  Win rate:           {pf['win_rate_pct']:.1f}%")
    print(f"  Profit factor:      {pf['profit_factor']:.3f}")
    print(f"  Sharpe / Sortino:   {pf['sharpe']:.2f} / {pf['sortino']:.2f}")
    print(f"  Calmar:             {pf['calmar']:.2f}")
    print(f"  N ejecutadas:       {pf['n_ejecutadas']}  (descartadas: {pf['n_descartadas']})")
    print(f"  Wins / Losses:      {pf['n_wins']} / {pf['n_losses']}")
    print(f"  Tiempos (s):        {res['tiempos']}")

    if args.guardar:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = REPORTS_DIR / f"backtest_gpu_{ts}.json"
        fn.write_text(json.dumps(res, default=str, indent=2), encoding="utf-8")
        print(f"\n  Guardado: {fn}")


if __name__ == "__main__":
    main()
