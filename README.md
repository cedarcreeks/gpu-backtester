# gpu-backtester

GPU-accelerated **point-in-time (PIT) equity backtester** with vectorized
OCO (one-cancels-other) trade simulation. Uses [CuPy](https://cupy.dev) for
NVIDIA GPU acceleration with a CPU fallback engine (Numba) for parity testing.

Built for nightly research runs over US equity universes (S&P 500 / S&P 400),
but the engine is generic enough for any OHLCV dataset.

## Why GPU?

A single walk-forward pass over a ~600-ticker universe × 4 years has two
expensive stages:

| Stage                     | CPU (Numba)      | GPU (CuPy)         |
|---------------------------|------------------|--------------------|
| RS rating + indicators    | seconds–minutes  | sub-second batches |
| OCO trade simulation      | per-trade loop   | `[K, max_hold+1]` vectorized step |

For large universes the GPU port is 10–30× faster on precompute and 30–50× on
sim. For small universes (< 20 tickers) the CPU engine wins because kernel
launch overhead dominates — the validator detects this.

## What it does

- Point-in-time universe (no survivorship bias) via Wikipedia S&P historical
  constituents + technical filters
- RS Rating cross-sectional (5 periods, configurable weights)
- Per-day candidate extraction (state classification, indicators)
- OCO trade simulation: entry, stop-loss, T1 partial close + breakeven shift,
  T2 target, trailing stop (% or chandelier ATR), max-hold expiration
- Portfolio simulation: position cap, risk-per-trade sizing, heat cap,
  drawdown / Calmar / Sharpe / Sortino / streaks

The GPU and CPU engines produce **functionally identical trades** — exact
parity validated end-to-end on a real PIT universe of 905 US equities ×
3.5 years (~77k trades, 0 diff on `pnl_pct`, 0 diff on outcome). Walk-forward
runs pick the same best params on every fold with identical out-of-sample
returns. See `gpu_bt/compare.py` to reproduce.

> Bit-for-bit identity is *not* possible because CuPy reorders sums across
> thread blocks. The validator checks trade-level equality (entry/exit/PnL/
> outcome) and metric tolerance 1e-9 — both pass exactly.

## Install

### Base (CPU only)

```bash
pip install -r requirements.txt
```

### With GPU support (NVIDIA, CUDA 12.x)

```bash
pip install -r requirements.txt -r requirements-gpu.txt
```

On Windows the NVIDIA wheels bundle the runtime DLLs (`nvrtc64_120_0.dll`,
etc.) and `gpu_bt/gpu.py` registers them automatically via
`os.add_dll_directory` before importing CuPy. No system-wide CUDA Toolkit
install required.

Sanity check:

```python
from gpu_bt import gpu_disponible
print(gpu_disponible())   # True if CuPy + CUDA device available
```

## Quickstart

```python
import datetime
from gpu_bt import (
    Params, gpu_disponible,
    historico_a_tensor, extract_candidatos_gpu, simulate_trades_gpu,
    filtrar_candidatos, simular_portfolio,
)
from gpu_bt.pit_universe import PitUniverse

# 1) Build point-in-time universe (downloads via yfinance, caches Parquet)
pit = PitUniverse()
pit.build("2022-01-01", "2024-12-31")
historico = pit.historico   # dict[ticker → DataFrame OHLCV]

# 2) Stack into GPU tensor [N_tickers, T_days, 5]
tensor = historico_a_tensor(historico)

# 3) Precompute candidates (RS, indicators, classification)
desde = datetime.date(2023, 1, 1)
hasta = datetime.date(2024, 12, 31)
candidatos = extract_candidatos_gpu(tensor, desde, hasta)

# 4) Apply parameter filters (CPU side, dict-only)
params  = Params(rs_minimo=70, score_minimo=60)
señales = filtrar_candidatos(candidatos, params)

# 5) Vectorized OCO simulation on GPU
operaciones = simulate_trades_gpu(señales, tensor, params)

# 6) Portfolio sizing + metrics (CPU)
pf = simular_portfolio(operaciones, params)
print(f"Capital final: ${pf['capital_final']:,.2f}  "
      f"WR={pf['win_rate_pct']}%  PF={pf['profit_factor']}  "
      f"Calmar={pf['calmar']}  MaxDD={pf['max_drawdown_pct']}%")
```

## CLI

```bash
# Single-period nightly run with PIT universe
python -m gpu_bt.backtest_gpu --desde 2022-01-01 --hasta 2024-12-31 --guardar

# Walk-forward grid search on GPU (5 windows × N param combos)
python -m gpu_bt.wf --desde 2022-01-01 --hasta 2025-06-30 \
                    --ventanas 5 --grid rapido --estado BREAKOUT --guardar

# CPU vs GPU comparison (timings + parity report)
python -m gpu_bt.compare --desde 2022-01-01 --hasta 2025-06-30

# Validate GPU output matches CPU (trades identical, metrics 1e-9)
python -m gpu_bt.validate --desde 2024-01-01 --hasta 2024-06-30
```

Reports are saved as JSON in `reports/`.

## Project layout

```
gpu_bt/
├── core.py          Pure CPU engine (Params, simulate, portfolio)
├── gpu.py           CuPy kernels (tensor stack, RS, indicators, batched OCO)
├── cache.py         Parquet OHLCV cache + Numba kernels for the CPU path
├── pit_universe.py  Historical S&P 500 / S&P 400 constituents (Wikipedia) + filters
├── wf.py            Walk-forward grid search on GPU
├── wf_cpu.py        Walk-forward grid search on CPU (reference)
├── backtest_gpu.py  Single-period CLI entry
├── compare.py       CPU vs GPU timing + parity comparison
└── validate.py      CPU vs GPU parity validator (smoke smoke)
```

## Customizing parameters

`Params` (in `gpu_bt.core`) is a `dataclass` exposing every threshold:

```python
Params(
    rs_minimo=50,            # min RS rating filter
    vol_minimo_m=0.5,        # min average dollar volume (millions)
    pct_max_52w=-15.0,       # min pct from 52w high (drop weak names)
    score_minimo=0.0,
    rvol_minimo=1.5,
    stop_pct=0.07,
    target_rr_1=2.0,
    target_rr_2=3.0,
    fraccion_t1=0.5,
    trailing_stop_pct=0.07,
    max_hold_days=20,
    max_posiciones=5,
    capital_total=10_000.0,
    riesgo_por_op=0.02,
    heat_cap=1.0,
    filtro_mercado=True,     # only enter when SPY > SMA200
    filtro_earnings=True,
)
```

Comments and identifiers in the codebase are partly in Spanish — translation
PRs welcome.

## Known caveats

- Bit-for-bit GPU/CPU identity is **not** possible: CuPy reorders sums across
  thread blocks, so the last decimal of any non-trivial float reduction
  diverges. The validator allows trade-level equality + metric tolerance 1e-9.
- The PIT universe relies on Wikipedia revision history for index
  constituents, which can lag actual index changes by a few days.
- Only daily-bar backtests are supported. Intraday adapters welcome.
- `cupyx.scipy.ndimage.maximum_filter1d` has an **inverted `origin` sign**
  versus SciPy. `gpu.py` uses `origin=+126` to get a `[t-252, t]` window.

## License

MIT — see [LICENSE](LICENSE).
