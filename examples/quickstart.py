"""
examples/quickstart.py — Minimal end-to-end backtest on synthetic data.

Run:
    python examples/quickstart.py
"""
import sys
import os
import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_bt import (
    Params, gpu_disponible,
    historico_a_tensor, extract_candidatos_gpu, simulate_trades_gpu,
    filtrar_candidatos, simular_portfolio,
)


def synthetic_history(n_tickers=10, n_days=600, seed=42):
    """Generate fake OHLCV for n_tickers across n_days."""
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rng_master = np.random.default_rng(seed)
    out = {}
    for i in range(n_tickers + 1):  # +1 for SPY
        rng = np.random.default_rng(rng_master.integers(0, 1_000_000))
        drift = rng.normal(0.0004, 0.0003)
        rets = rng.normal(drift, 0.02, n_days)
        close = 100 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low  = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        op   = close * (1 + rng.normal(0, 0.003, n_days))
        vol  = rng.integers(500_000, 5_000_000, n_days).astype(float)
        key = "SPY" if i == n_tickers else f"T{i:02d}"
        out[key] = pd.DataFrame(
            {"Open": op, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=dates,
        )
    return out


def main():
    if not gpu_disponible():
        print("GPU not available — install requirements-gpu.txt for the GPU path.")
        return

    print("Generating synthetic OHLCV...")
    historico = synthetic_history(n_tickers=20, n_days=600)

    print("Stacking tensor and extracting candidates on GPU...")
    tensor = historico_a_tensor(historico)
    candidatos = extract_candidatos_gpu(
        tensor,
        datetime.date(2023, 1, 1),
        datetime.date(2024, 6, 1),
    )
    print(f"  {len(candidatos)} candidates")

    params = Params(rs_minimo=50, score_minimo=0.0)
    señales = filtrar_candidatos(candidatos, params)
    print(f"  {len(señales)} signals after filters")

    operaciones = simulate_trades_gpu(señales, tensor, params)
    pf = simular_portfolio(operaciones, params)
    print()
    print(f"  Capital final:    ${pf['capital_final']:,.2f}")
    print(f"  Return:           {pf['retorno_pct']:+.2f}%")
    print(f"  Max DD:           {pf['max_drawdown_pct']:.2f}%")
    print(f"  Win rate:         {pf['win_rate_pct']:.1f}%")
    print(f"  Profit factor:    {pf['profit_factor']:.3f}")
    print(f"  N trades:         {pf['n_ejecutadas']}")


if __name__ == "__main__":
    main()
