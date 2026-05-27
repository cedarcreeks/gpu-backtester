"""
validate_gpu_cpu.py — Validador paridad CPU vs GPU
===============================================================================
Corre el mismo backtest PIT en ambos motores y compara:
  - Candidatos (mismo set, mismo orden, mismos campos)
  - Operaciones (mismo set, mismo outcome/pnl/dias por trade)
  - Métricas portfolio (con tolerancia 1e-9)

Falla con exit 1 si hay divergencia. Útil para nightly post-deploy.

Uso:
  python scripts/validate_gpu_cpu.py --desde 2024-01-01 --hasta 2024-06-30
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
import time

from gpu_bt.core import (
    Params, precomputar_candidatos, filtrar_candidatos,
    simular_operacion, simular_portfolio,
)
from gpu_bt.gpu import (
    gpu_disponible, historico_a_tensor, extract_candidatos_gpu,
    simulate_trades_gpu,
)
from gpu_bt.backtest_gpu import cargar_pit


def parse_date(s): return datetime.date.fromisoformat(s)

TOL = 1e-9


def diff_metric(name, a, b, tol=TOL):
    """Devuelve None si OK, str con detalle si diverge."""
    if a is None and b is None:
        return None
    if (a is None) ^ (b is None):
        return f"{name}: CPU={a} GPU={b}"
    if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
        if a != b:
            return f"{name}: CPU={a} GPU={b}"
        return None
    try:
        if abs(float(a) - float(b)) > tol * max(1.0, abs(float(a))):
            return f"{name}: CPU={a} GPU={b} (diff={float(a)-float(b):.3e})"
    except Exception:
        if a != b:
            return f"{name}: CPU={a!r} GPU={b!r}"
    return None


def compara_candidatos(cands_cpu, cands_gpu) -> list:
    errs = []
    key = lambda c: (c["fecha"], c["ticker"])
    cpu_map = {key(c): c for c in cands_cpu}
    gpu_map = {key(c): c for c in cands_gpu}

    only_cpu = set(cpu_map) - set(gpu_map)
    only_gpu = set(gpu_map) - set(cpu_map)
    if only_cpu:
        errs.append(f"[cands] solo en CPU ({len(only_cpu)}): {list(only_cpu)[:5]}...")
    if only_gpu:
        errs.append(f"[cands] solo en GPU ({len(only_gpu)}): {list(only_gpu)[:5]}...")

    shared = set(cpu_map) & set(gpu_map)
    n_shown = 0
    fields_compare = ["rs", "p1w", "p1m", "p3m", "pct52", "rel_vol",
                      "calidad_vela", "estado", "regimen",
                      "mercado_alcista", "precio_señal", "entrada_real"]
    for k in shared:
        a = cpu_map[k]; b = gpu_map[k]
        for f in fields_compare:
            d = diff_metric(f, a.get(f), b.get(f))
            if d is not None and n_shown < 30:
                errs.append(f"[cand {k}] {d}")
                n_shown += 1
    return errs


def compara_operaciones(ops_cpu, ops_gpu) -> list:
    errs = []
    key = lambda o: (o["fecha_entrada"], o["ticker"])
    cpu_map = {key(o): o for o in ops_cpu}
    gpu_map = {key(o): o for o in ops_gpu}

    only_cpu = set(cpu_map) - set(gpu_map)
    only_gpu = set(gpu_map) - set(cpu_map)
    if only_cpu:
        errs.append(f"[ops] solo en CPU ({len(only_cpu)})")
    if only_gpu:
        errs.append(f"[ops] solo en GPU ({len(only_gpu)})")

    shared = set(cpu_map) & set(gpu_map)
    n_shown = 0
    for k in shared:
        a = cpu_map[k]; b = gpu_map[k]
        for f in ["outcome", "motivo_salida", "t1_alcanzado",
                  "dias_en_operacion", "precio_salida", "pnl_pct"]:
            d = diff_metric(f, a.get(f), b.get(f), tol=1e-6)
            if d is not None and n_shown < 30:
                errs.append(f"[op {k}] {d}")
                n_shown += 1
    return errs


def compara_portfolio(pf_cpu, pf_gpu) -> list:
    errs = []
    for f in ("capital_final", "retorno_pct", "max_drawdown_pct",
              "win_rate_pct", "profit_factor", "sharpe", "sortino",
              "calmar", "n_ejecutadas", "n_wins", "n_losses"):
        d = diff_metric(f, pf_cpu.get(f), pf_gpu.get(f), tol=1e-6)
        if d is not None:
            errs.append(f"[pf] {d}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=parse_date,
                    default=datetime.date(2024, 1, 1))
    ap.add_argument("--hasta", type=parse_date,
                    default=datetime.date(2024, 6, 30))
    ap.add_argument("--estado", type=str, default=None)
    args = ap.parse_args()

    if not gpu_disponible():
        print("ERROR: GPU no disponible.")
        _sys.exit(1)

    params = Params()
    historico = cargar_pit(args.desde, args.hasta)

    print(f"[CPU] precompute candidatos...")
    t0 = time.time()
    cands_cpu = precomputar_candidatos(historico, args.desde, args.hasta,
                                        n_workers=1)
    t_cpu_pre = time.time() - t0
    print(f"[CPU] {len(cands_cpu)} candidatos en {t_cpu_pre:.1f}s")

    print(f"[GPU] precompute candidatos...")
    t0 = time.time()
    tensor = historico_a_tensor(historico)
    cands_gpu = extract_candidatos_gpu(tensor, args.desde, args.hasta)
    t_gpu_pre = time.time() - t0
    print(f"[GPU] {len(cands_gpu)} candidatos en {t_gpu_pre:.1f}s "
          f"({t_cpu_pre/max(t_gpu_pre,0.01):.1f}x)")

    errs_c = compara_candidatos(cands_cpu, cands_gpu)
    if errs_c:
        print(f"\n[FAIL candidatos] {len(errs_c)} divergencias:")
        for e in errs_c[:20]:
            print(" ", e)
    else:
        print("[OK candidatos]")

    señales_cpu = filtrar_candidatos(cands_cpu, params, args.estado)
    señales_gpu = filtrar_candidatos(cands_gpu, params, args.estado)
    print(f"\nseñales CPU={len(señales_cpu)} GPU={len(señales_gpu)}")

    # Sim
    print(f"[CPU] sim trades...")
    t0 = time.time()
    ops_cpu = [simular_operacion(s, historico, params) for s in señales_cpu]
    t_cpu_sim = time.time() - t0
    print(f"[CPU] {len(ops_cpu)} ops en {t_cpu_sim:.1f}s")

    print(f"[GPU] sim trades...")
    t0 = time.time()
    ops_gpu = simulate_trades_gpu(señales_gpu, tensor, params)
    t_gpu_sim = time.time() - t0
    print(f"[GPU] {len(ops_gpu)} ops en {t_gpu_sim:.1f}s "
          f"({t_cpu_sim/max(t_gpu_sim,0.01):.1f}x)")

    errs_o = compara_operaciones(ops_cpu, ops_gpu)
    if errs_o:
        print(f"\n[FAIL ops] {len(errs_o)} divergencias:")
        for e in errs_o[:20]:
            print(" ", e)
    else:
        print("[OK ops]")

    pf_cpu = simular_portfolio(ops_cpu, params)
    pf_gpu = simular_portfolio(ops_gpu, params)

    errs_p = compara_portfolio(pf_cpu, pf_gpu)
    if errs_p:
        print(f"\n[FAIL portfolio] {len(errs_p)} divergencias:")
        for e in errs_p:
            print(" ", e)
    else:
        print("[OK portfolio]")

    total_errs = len(errs_c) + len(errs_o) + len(errs_p)
    print(f"\n{'═'*60}")
    print(f"  Total divergencias: {total_errs}")
    print(f"  Speedup precompute: {t_cpu_pre/max(t_gpu_pre,0.01):.1f}x")
    print(f"  Speedup sim:        {t_cpu_sim/max(t_gpu_sim,0.01):.1f}x")
    print(f"{'═'*60}")
    _sys.exit(0 if total_errs == 0 else 1)


if __name__ == "__main__":
    main()
