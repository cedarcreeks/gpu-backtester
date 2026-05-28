"""
backtest_wf_gpu.py — Walk-Forward Grid Search en GPU (CuPy).

Replica del CPU engine pero con kernels GPU:
  - Por cada ventana: precompute GPU una vez (train + test)
  - Para cada params del grid: filtrar (CPU) + simulate_trades_gpu + portfolio
  - Pick best train → eval test

Usa build_ventanas + build_param_grid + _score_resultado + print_reporte_wf
del modulo CPU para garantizar parity de criterios.
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

from gpu_bt.core import (
    Params, filtrar_candidatos, simular_portfolio,
)
from gpu_bt.wf_cpu import (
    build_param_grid, build_ventanas, _score_resultado,
    print_reporte_wf, guardar_resultados_wf,
)
from gpu_bt.gpu import (
    gpu_disponible, historico_a_tensor, extract_candidatos_gpu,
    simulate_trades_gpu,
)


def _eval_params_gpu(params, cands, tensor, estado_filtro=None):
    señales = filtrar_candidatos(cands, params, estado_filtro)
    if not señales:
        return {
            "params":        params.as_dict(),
            "params_label":  params.label(),
            "n_ops":         0, "wr_pct": 0, "retorno_pct": 0,
            "max_dd_pct":    0, "profit_factor": 0,
        }
    ops = simulate_trades_gpu(señales, tensor, params)
    pf = simular_portfolio(ops, params)
    if pf.get("error"):
        return {
            "params":        params.as_dict(),
            "params_label":  params.label(),
            "n_ops":         0, "wr_pct": 0, "retorno_pct": 0,
            "max_dd_pct":    0, "profit_factor": 0,
        }
    return {
        "params":        params.as_dict(),
        "params_label":  params.label(),
        "n_ops":         pf["n_ejecutadas"],
        "wr_pct":        pf["win_rate_pct"],
        "retorno_pct":   pf["retorno_pct"],
        "max_dd_pct":    pf["max_drawdown_pct"],
        "profit_factor": pf["profit_factor"],
        "sharpe":        pf.get("sharpe", 0),
        "sortino":       pf.get("sortino", 0),
        "calmar":        pf.get("calmar", 0),
    }


def run_walk_forward_gpu(historico, desde, hasta,
                          n_ventanas=5, ratio_train=0.7,
                          param_grid=None, estado_filtro=None,
                          verbose=True):
    grid = param_grid or build_param_grid("normal")
    ventanas = build_ventanas(desde, hasta, n_ventanas, ratio_train)
    if not ventanas:
        return {"error": "No se pudieron generar ventanas"}

    if verbose:
        print("\n" + "═" * 70)
        print("  WALK-FORWARD GPU")
        print("═" * 70)
        print(f"  Período:        {desde} → {hasta}")
        print(f"  Ventanas:       {len(ventanas)}")
        print(f"  Grid params:    {len(grid)} combinaciones")
        print(f"  Estado filtro:  {estado_filtro or 'todos'}")
        for v in ventanas:
            print(f"  V{v['ventana']}: Train {v['train_desde']}→{v['train_hasta']}"
                  f"   Test {v['test_desde']}→{v['test_hasta']}")
        print()

    t0_total = time.time()
    print(f"[WF-GPU] stack tensor...")
    t0 = time.time()
    tensor = historico_a_tensor(historico)
    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    arr = tensor["arr_gpu"]
    N, T, _ = arr.shape
    print(f"[WF-GPU] tensor [{N},{T},5] en {time.time()-t0:.1f}s")

    resultados_ventanas = []
    for v in ventanas:
        vid = v["ventana"]
        if verbose:
            print(f"\n  [V{vid}] precompute train...", end=" ", flush=True)
        t0 = time.time()
        cands_train = extract_candidatos_gpu(tensor, v["train_desde"], v["train_hasta"])
        cp.cuda.Stream.null.synchronize()
        if verbose:
            print(f"{len(cands_train)} cands {time.time()-t0:.1f}s")
            print(f"  [V{vid}] precompute test...", end=" ", flush=True)
        t0 = time.time()
        cands_test = extract_candidatos_gpu(tensor, v["test_desde"], v["test_hasta"])
        cp.cuda.Stream.null.synchronize()
        if verbose:
            print(f"{len(cands_test)} cands {time.time()-t0:.1f}s")
            print(f"  [V{vid}] TRAIN {len(grid)} params...", end=" ", flush=True)
        t0 = time.time()
        train_results = [_eval_params_gpu(p, cands_train, tensor, estado_filtro)
                         for p in grid]
        cp.cuda.Stream.null.synchronize()
        if verbose:
            print(f"✓ {time.time()-t0:.1f}s")

        validos = [r for r in train_results if r["n_ops"] >= 3]
        if not validos:
            best_train = {
                "params":        Params().as_dict(),
                "params_label":  Params().label(),
                "retorno_pct":   0, "max_dd_pct": 0,
                "profit_factor": 0, "n_ops":      0, "wr_pct": 0,
            }
        else:
            best_train = max(validos, key=_score_resultado)

        # Eval test con best train
        best_params = Params(**{k: v for k, v in best_train["params"].items()
                                if k in Params.__dataclass_fields__})
        test_result = _eval_params_gpu(best_params, cands_test, tensor, estado_filtro)

        if verbose:
            print(f"  [V{vid}] best train: {best_train['params_label']}  "
                  f"ret={best_train['retorno_pct']:+.1f}% dd={best_train['max_dd_pct']:.1f}% "
                  f"pf={best_train['profit_factor']:.2f} n={best_train['n_ops']}")
            print(f"  [V{vid}] TEST:        "
                  f"ret={test_result['retorno_pct']:+.1f}% dd={test_result['max_dd_pct']:.1f}% "
                  f"pf={test_result['profit_factor']:.2f} n={test_result['n_ops']}")

        resultados_ventanas.append({
            "ventana":      vid,
            "train_desde":  str(v["train_desde"]),
            "train_hasta":  str(v["train_hasta"]),
            "test_desde":   str(v["test_desde"]),
            "test_hasta":   str(v["test_hasta"]),
            "best_train":   best_train,
            "test":         test_result,
            "all_train":    train_results,
        })

    elapsed_total = time.time() - t0_total

    # Stats agregadas (oos = out-of-sample = test)
    test_rets = [r["test"]["retorno_pct"] for r in resultados_ventanas]
    test_wrs  = [r["test"]["wr_pct"]      for r in resultados_ventanas]
    test_pfs  = [r["test"]["profit_factor"] for r in resultados_ventanas]
    test_dds  = [r["test"]["max_dd_pct"]  for r in resultados_ventanas]
    import statistics
    summary = {
        "n_ventanas":       len(resultados_ventanas),
        "elapsed_total_s":  round(elapsed_total, 1),
        "oos_mean_ret_pct": round(statistics.mean(test_rets) if test_rets else 0, 2),
        "oos_median_ret_pct": round(statistics.median(test_rets) if test_rets else 0, 2),
        "oos_mean_wr_pct":  round(statistics.mean(test_wrs) if test_wrs else 0, 2),
        "oos_mean_pf":      round(statistics.mean(test_pfs) if test_pfs else 0, 3),
        "oos_max_dd_pct":   round(max(test_dds) if test_dds else 0, 2),
        "oos_positive_pct": round(sum(1 for r in test_rets if r > 0) / len(test_rets) * 100, 1)
                            if test_rets else 0,
    }
    if verbose:
        print(f"\n  TOTAL WF-GPU: {elapsed_total:.1f}s")
        print(f"  OOS mean ret:  {summary['oos_mean_ret_pct']:+.2f}%")
        print(f"  OOS median ret:{summary['oos_median_ret_pct']:+.2f}%")
        print(f"  OOS mean WR:   {summary['oos_mean_wr_pct']:.1f}%")
        print(f"  OOS mean PF:   {summary['oos_mean_pf']:.3f}")
        print(f"  OOS max DD:    {summary['oos_max_dd_pct']:.1f}%")
        print(f"  Ventanas positivas: {summary['oos_positive_pct']:.1f}%")

    return {
        "ventanas":  resultados_ventanas,
        "summary":   summary,
        "n_params":  len(grid),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=lambda s: datetime.date.fromisoformat(s),
                    default=datetime.date(2022, 1, 1))
    ap.add_argument("--hasta", type=lambda s: datetime.date.fromisoformat(s),
                    default=datetime.date(2025, 6, 30))
    ap.add_argument("--ventanas", type=int, default=5)
    ap.add_argument("--ratio-train", type=float, default=0.7)
    ap.add_argument("--grid", choices=["rapido","normal","completo"], default="normal")
    ap.add_argument("--estado", type=str, default=None)
    ap.add_argument("--filtro-mercado", action="store_true", default=False,
                    help="Solo combinaciones con SPY>SMA200 (filtro_mercado=True)")
    ap.add_argument("--guardar", action="store_true")
    args = ap.parse_args()

    if not gpu_disponible():
        print("ERROR: GPU no disponible.")
        return

    from gpu_bt.pit_universe import PitUniverse
    print(f"[WF-GPU] Cargando PIT...")
    pit = PitUniverse()
    pit.build(str(args.desde), str(args.hasta))
    historico = pit.historico
    print(f"[WF-GPU] PIT: {len(historico)} tickers")

    grid = build_param_grid(args.grid)
    if args.filtro_mercado:
        antes = len(grid)
        grid = [p for p in grid if p.filtro_mercado]
        print(f"[WF-GPU] --filtro-mercado: {antes} → {len(grid)} params")
    print(f"[WF-GPU] Grid {args.grid}: {len(grid)} params")

    res = run_walk_forward_gpu(
        historico, args.desde, args.hasta,
        n_ventanas=args.ventanas, ratio_train=args.ratio_train,
        param_grid=grid, estado_filtro=args.estado,
    )

    if args.guardar:
        rep_dir = pathlib.Path("reports")
        rep_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Strip equity_curve para JSON sano
        for vw in res.get("ventanas", []):
            for k in ("best_train", "test"):
                if isinstance(vw.get(k), dict) and "equity_curve" in vw[k]:
                    del vw[k]["equity_curve"]
        fn = rep_dir / f"wf_gpu_{args.grid}_{ts}.json"
        fn.write_text(json.dumps(res, default=str, indent=2), encoding="utf-8")
        print(f"\n  Guardado: {fn}")


if __name__ == "__main__":
    main()
