"""
gpu_bt — GPU-accelerated PIT backtester (CuPy).
"""
__version__ = "0.1.0"

from gpu_bt.core import (
    Params,
    precomputar_candidatos,
    filtrar_candidatos,
    simular_operacion,
    simular_portfolio,
    run_backtest,
    run_backtest_rapido,
)

from gpu_bt.gpu import (
    gpu_disponible,
    historico_a_tensor,
    compute_perfs_gpu,
    compute_rs_gpu,
    compute_indicators_gpu,
    extract_candidatos_gpu,
    simulate_trades_gpu,
)

from gpu_bt.wf import run_walk_forward_gpu

__all__ = [
    "Params",
    "precomputar_candidatos",
    "filtrar_candidatos",
    "simular_operacion",
    "simular_portfolio",
    "run_backtest",
    "run_backtest_rapido",
    "gpu_disponible",
    "historico_a_tensor",
    "compute_perfs_gpu",
    "compute_rs_gpu",
    "compute_indicators_gpu",
    "extract_candidatos_gpu",
    "simulate_trades_gpu",
    "run_walk_forward_gpu",
]
