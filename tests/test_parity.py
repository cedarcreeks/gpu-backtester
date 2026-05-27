"""
tests/test_parity.py — CPU vs GPU functional parity on synthetic data.

Run:
    python -m pytest tests/ -v
"""
import datetime
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_bt import (
    Params, gpu_disponible,
    historico_a_tensor, extract_candidatos_gpu, simulate_trades_gpu,
    precomputar_candidatos, filtrar_candidatos, simular_operacion,
    simular_portfolio,
)


@pytest.fixture
def historico():
    dates = pd.bdate_range("2022-01-03", periods=600)
    np.random.seed(42)

    def gen(seed, drift=0.0003):
        rng = np.random.default_rng(seed)
        rets = rng.normal(drift, 0.02, len(dates))
        close = 100 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.005, len(dates))))
        low  = close * (1 - np.abs(rng.normal(0, 0.005, len(dates))))
        op   = close * (1 + rng.normal(0, 0.003, len(dates)))
        vol  = rng.integers(500_000, 5_000_000, len(dates)).astype(float)
        return pd.DataFrame(
            {"Open": op, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=dates,
        )

    h = {f"T{i:02d}": gen(i, 0.0003 + (i % 5) * 0.0002) for i in range(15)}
    h["SPY"] = gen(99, 0.0003)
    return h


@pytest.mark.skipif(not gpu_disponible(), reason="GPU unavailable")
def test_trade_parity(historico):
    params = Params()
    desde = datetime.date(2023, 1, 1)
    hasta = datetime.date(2024, 6, 1)

    cands_cpu = precomputar_candidatos(historico, desde, hasta, n_workers=1)
    señales = filtrar_candidatos(cands_cpu, params)

    tensor = historico_a_tensor(historico)
    ops_cpu = [simular_operacion(s, historico, params) for s in señales]
    ops_gpu = simulate_trades_gpu(señales, tensor, params)

    assert len(ops_cpu) == len(ops_gpu)
    diffs = 0
    for a, b in zip(ops_cpu, ops_gpu):
        if abs(a["pnl_pct"] - b["pnl_pct"]) > 1e-6:
            diffs += 1
    assert diffs == 0, f"{diffs} GPU trades diverge from CPU"


@pytest.mark.skipif(not gpu_disponible(), reason="GPU unavailable")
def test_portfolio_parity(historico):
    params = Params()
    desde = datetime.date(2023, 1, 1)
    hasta = datetime.date(2024, 6, 1)

    cands_cpu = precomputar_candidatos(historico, desde, hasta, n_workers=1)
    señales = filtrar_candidatos(cands_cpu, params)

    tensor = historico_a_tensor(historico)
    ops_cpu = [simular_operacion(s, historico, params) for s in señales]
    ops_gpu = simulate_trades_gpu(señales, tensor, params)

    pf_cpu = simular_portfolio(ops_cpu, params)
    pf_gpu = simular_portfolio(ops_gpu, params)

    for key in ("capital_final", "win_rate_pct", "profit_factor",
                "n_ejecutadas", "n_wins", "n_losses",
                "max_drawdown_pct"):
        assert pf_cpu[key] == pf_gpu[key], f"{key}: CPU={pf_cpu[key]} GPU={pf_gpu[key]}"
