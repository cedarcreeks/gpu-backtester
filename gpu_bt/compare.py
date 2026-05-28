"""
compare_cpu_gpu_pit.py — Comparación timings + métricas CPU vs GPU sobre PIT real.

Sin overfitting:
  - Params default (no grid search tuning)
  - Mismo período + mismo universo PIT en ambos
  - Single backtest (no in-sample tuning)
"""
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
import multiprocessing


def parse_date(s):
    return datetime.date.fromisoformat(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=parse_date, default=datetime.date(2022, 1, 1))
    ap.add_argument("--hasta", type=parse_date, default=datetime.date(2025, 6, 30))
    ap.add_argument("--workers", type=int, default=None,
                    help="Workers CPU para precompute (default: cpu_count)")
    ap.add_argument("--estado", type=str, default=None,
                    help="Filtro estado: BREAKOUT/SQUEEZE/EN BASE/EXTENDIDO/VIGILANCIA")
    args = ap.parse_args()

    from gpu_bt.core import (
        Params, precomputar_candidatos, filtrar_candidatos,
        simular_operacion, simular_portfolio,
    )
    from gpu_bt.gpu import (
        gpu_disponible, historico_a_tensor, extract_candidatos_gpu,
        simulate_trades_gpu,
    )
    from gpu_bt.pit_universe import PitUniverse

    if not gpu_disponible():
        print("ERROR: GPU no disponible.")
        _sys.exit(1)

    n_workers = args.workers or multiprocessing.cpu_count()

    print("═" * 75)
    print(f"  CPU vs GPU PIT backtest")
    print(f"  Período: {args.desde} → {args.hasta}")
    print(f"  Workers CPU: {n_workers}")
    print("═" * 75)

    # ─── Cargar PIT ────────────────────────────────────────────────────────
    print("\n[1/4] Cargando universo PIT...")
    t0 = time.time()
    pit = PitUniverse()
    pit.build(str(args.desde), str(args.hasta))
    historico = pit.historico
    t_pit = time.time() - t0
    n_tickers = len([k for k in historico if k != "SPY" and not k.startswith("ETF_")])
    print(f"  {len(historico)} tickers totales ({n_tickers} no-SPY/ETF) en {t_pit:.1f}s")

    params = Params()  # defaults — sin tuning

    # ═══════════════════════════════════════════════════════════════════════
    #  CPU PATH
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/4] CPU backtest...")

    t0 = time.time()
    cands_cpu = precomputar_candidatos(historico, args.desde, args.hasta,
                                        n_workers=n_workers)
    t_cpu_pre = time.time() - t0
    print(f"  precompute: {len(cands_cpu)} cands en {t_cpu_pre:.1f}s ({n_workers} workers)")

    t0 = time.time()
    señales_cpu = filtrar_candidatos(cands_cpu, params, args.estado)
    t_cpu_filt = time.time() - t0
    print(f"  filtrar:    {len(señales_cpu)} señales en {t_cpu_filt:.2f}s")

    t0 = time.time()
    ops_cpu = [simular_operacion(s, historico, params) for s in señales_cpu]
    t_cpu_sim = time.time() - t0
    print(f"  sim:        {len(ops_cpu)} ops en {t_cpu_sim:.1f}s")

    t0 = time.time()
    pf_cpu = simular_portfolio(ops_cpu, params)
    t_cpu_pf = time.time() - t0
    print(f"  portfolio:  {t_cpu_pf:.2f}s")
    t_cpu_total = t_cpu_pre + t_cpu_filt + t_cpu_sim + t_cpu_pf
    print(f"  TOTAL CPU:  {t_cpu_total:.1f}s")

    # ═══════════════════════════════════════════════════════════════════════
    #  GPU PATH
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/4] GPU backtest...")

    t0 = time.time()
    tensor = historico_a_tensor(historico)
    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    t_gpu_stack = time.time() - t0
    arr = tensor["arr_gpu"]
    N, T, _ = arr.shape
    print(f"  stack tensor [{N},{T},5] en {t_gpu_stack:.1f}s")

    t0 = time.time()
    cands_gpu = extract_candidatos_gpu(tensor, args.desde, args.hasta)
    cp.cuda.Stream.null.synchronize()
    t_gpu_pre = time.time() - t0
    print(f"  precompute: {len(cands_gpu)} cands en {t_gpu_pre:.1f}s")

    t0 = time.time()
    señales_gpu = filtrar_candidatos(cands_gpu, params, args.estado)
    t_gpu_filt = time.time() - t0
    print(f"  filtrar:    {len(señales_gpu)} señales en {t_gpu_filt:.2f}s")

    t0 = time.time()
    ops_gpu = simulate_trades_gpu(señales_gpu, tensor, params)
    cp.cuda.Stream.null.synchronize()
    t_gpu_sim = time.time() - t0
    print(f"  sim:        {len(ops_gpu)} ops en {t_gpu_sim:.1f}s")

    t0 = time.time()
    pf_gpu = simular_portfolio(ops_gpu, params)
    t_gpu_pf = time.time() - t0
    print(f"  portfolio:  {t_gpu_pf:.2f}s")
    t_gpu_total = t_gpu_stack + t_gpu_pre + t_gpu_filt + t_gpu_sim + t_gpu_pf
    print(f"  TOTAL GPU:  {t_gpu_total:.1f}s")

    # ═══════════════════════════════════════════════════════════════════════
    #  COMPARACIÓN
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/4] Comparación")
    print("─" * 75)
    print(f"  {'Stage':<20} {'CPU':>12} {'GPU':>12} {'Speedup':>10}")
    print("─" * 75)
    print(f"  {'precompute':<20} {t_cpu_pre:>10.1f}s  {t_gpu_pre + t_gpu_stack:>10.1f}s  {t_cpu_pre / max(t_gpu_pre + t_gpu_stack, 0.01):>8.1f}x")
    print(f"  {'sim trades':<20} {t_cpu_sim:>10.1f}s  {t_gpu_sim:>10.1f}s  {t_cpu_sim / max(t_gpu_sim, 0.01):>8.1f}x")
    print(f"  {'TOTAL':<20} {t_cpu_total:>10.1f}s  {t_gpu_total:>10.1f}s  {t_cpu_total / max(t_gpu_total, 0.01):>8.1f}x")
    print("─" * 75)

    print("\n  Métricas portfolio")
    print("─" * 75)
    print(f"  {'Metric':<22} {'CPU':>15} {'GPU':>15} {'Diff':>15}")
    print("─" * 75)
    for k in ("capital_final", "retorno_pct", "max_drawdown_pct",
              "win_rate_pct", "profit_factor", "sharpe", "sortino",
              "calmar", "n_ejecutadas", "n_wins", "n_losses",
              "max_win_streak", "max_loss_streak"):
        a = pf_cpu.get(k)
        b = pf_gpu.get(k)
        try:
            diff = float(a) - float(b)
            diff_str = f"{diff:+.4g}"
        except Exception:
            diff_str = "—" if a == b else "DIFF"
        print(f"  {k:<22} {str(a):>15} {str(b):>15} {diff_str:>15}")
    print("─" * 75)

    # Diff ops trade-level
    key = lambda o: (o["fecha_entrada"], o["ticker"])
    cpu_map = {key(o): o for o in ops_cpu}
    gpu_map = {key(o): o for o in ops_gpu}
    common = set(cpu_map) & set(gpu_map)
    only_cpu = set(cpu_map) - set(gpu_map)
    only_gpu = set(gpu_map) - set(cpu_map)
    n_diff_pnl = 0
    n_diff_outcome = 0
    for k in common:
        a = cpu_map[k]; b = gpu_map[k]
        if abs(a.get("pnl_pct", 0) - b.get("pnl_pct", 0)) > 1e-6:
            n_diff_pnl += 1
        if a.get("outcome") != b.get("outcome"):
            n_diff_outcome += 1
    print(f"\n  Trade-level parity:")
    print(f"    ops CPU:       {len(ops_cpu)}")
    print(f"    ops GPU:       {len(ops_gpu)}")
    print(f"    common:        {len(common)}")
    print(f"    solo CPU:      {len(only_cpu)}")
    print(f"    solo GPU:      {len(only_gpu)}")
    print(f"    diff pnl_pct:  {n_diff_pnl}")
    print(f"    diff outcome:  {n_diff_outcome}")

    # Save report
    rep_dir = pathlib.Path("reports")
    rep_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "periodo": {"desde": str(args.desde), "hasta": str(args.hasta)},
        "n_tickers": n_tickers,
        "tensor_shape": [N, T, 5],
        "tiempos_cpu": {
            "precompute_s": round(t_cpu_pre, 2),
            "filtrar_s":    round(t_cpu_filt, 2),
            "sim_s":        round(t_cpu_sim, 2),
            "portfolio_s":  round(t_cpu_pf, 2),
            "total_s":      round(t_cpu_total, 2),
        },
        "tiempos_gpu": {
            "stack_s":      round(t_gpu_stack, 2),
            "precompute_s": round(t_gpu_pre, 2),
            "filtrar_s":    round(t_gpu_filt, 2),
            "sim_s":        round(t_gpu_sim, 2),
            "portfolio_s":  round(t_gpu_pf, 2),
            "total_s":      round(t_gpu_total, 2),
        },
        "speedup": {
            "precompute": round(t_cpu_pre / max(t_gpu_pre + t_gpu_stack, 0.01), 2),
            "sim":        round(t_cpu_sim / max(t_gpu_sim, 0.01), 2),
            "total":      round(t_cpu_total / max(t_gpu_total, 0.01), 2),
        },
        "metricas_cpu": pf_cpu,
        "metricas_gpu": pf_gpu,
        "parity": {
            "n_ops_cpu":      len(ops_cpu),
            "n_ops_gpu":      len(ops_gpu),
            "common":         len(common),
            "solo_cpu":       len(only_cpu),
            "solo_gpu":       len(only_gpu),
            "diff_pnl":       n_diff_pnl,
            "diff_outcome":   n_diff_outcome,
        },
    }
    # equity curves not JSON-friendly tuples → strip
    if "equity_curve" in report["metricas_cpu"]:
        del report["metricas_cpu"]["equity_curve"]
    if "equity_curve" in report["metricas_gpu"]:
        del report["metricas_gpu"]["equity_curve"]
    fn = rep_dir / f"compare_cpu_gpu_{ts}.json"
    fn.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
    print(f"\n  Reporte: {fn}")


if __name__ == "__main__":
    main()
