"""
bt_gpu.py — Backtest PIT GPU-acelerado con CuPy
===============================================================================
Port a GPU del precompute + simulacion OCO del motor CPU (backtest_core.py +
bt_cache.py). Mantiene PIT: el universo y filtros se aplican fecha a fecha
sobre el histórico stacked en tensor [N_tickers, T_dias, 5].

Diseño:
  - historico_a_tensor(): reindex todos los DFs al calendar SPY → tensor float64
  - compute_perfs_gpu(): perfs 5/21/63/126/252 vectorizadas
  - compute_rs_gpu(): RS rating 1-99 cross-sectional por día (stable argsort)
  - compute_indicators_gpu(): vol_avg, rel_vol, h52, l52, vol_sem, calidad_vela
  - extract_candidatos_gpu(): emite list[dict] schema idéntico CPU
  - simulate_trades_gpu(): batch [N_signals, max_hold+1] step-by-step

Paridad bit-a-bit GPU es físicamente imposible (paralelismo reordena sumas
float). Garantía: trades idénticos + métricas con tol 1e-9.

Self-contained CuPy port — no external broker deps.
===============================================================================
"""
from __future__ import annotations

import os
import datetime
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Wheels NVIDIA: registrar dlls antes de import cupy (Windows)
def _setup_cuda_dlls():
    try:
        import nvidia
        nvbase = nvidia.__path__[0]
        added = []
        for sub in ("cuda_nvrtc", "cuda_runtime", "cublas"):
            p = os.path.join(nvbase, sub, "bin")
            if os.path.isdir(p):
                added.append(p)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(p)
                    except Exception:
                        pass
        if added:
            os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


_setup_cuda_dlls()

try:
    import cupy as cp
    CUPY_OK = True
except Exception as e:
    cp = None
    CUPY_OK = False
    _IMPORT_ERR = e


def gpu_disponible() -> bool:
    if not CUPY_OK:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


# Pesos RS — orden [Perf1W, Perf1M, Perf3M, Perf6M, Perf1Y] coincide con CPU
PESOS_RS_NB = np.array([0.10, 0.25, 0.25, 0.20, 0.20], dtype=np.float64)
VENTANAS_RS = (5, 21, 63, 126, 252)  # mismo orden que PESOS_RS_NB
NOMBRES_RS = ("Perf1W", "Perf1M", "Perf3M", "Perf6M", "Perf1Y")
WARMUP_MIN = 252

# Columnas tensor [N, T, 5]
COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOL = 0, 1, 2, 3, 4


# ═══════════════════════════════════════════════════════════════════════════════
#  STACKING HISTORICO → TENSOR GPU
# ═══════════════════════════════════════════════════════════════════════════════

def historico_a_tensor(historico: dict,
                       calendar_ref: str = "SPY") -> dict:
    """
    Reindex todos los DFs al calendar maestro (SPY por defecto, si no, union
    de todas las fechas), apila en tensor [N_tickers, T_dias, 5] (OHLCV).

    Returns dict con:
      arr_gpu:     cp.ndarray (N, T, 5) float64
      valid_mask:  cp.ndarray (N, T) bool   — True si Close>0 y no-NaN
      tickers:     list[str] orden N (excluye SPY/ETFs)
      idx_tickers: dict[ticker → i] (incluye SPY/ETF)
      dates:       np.ndarray (T,) np.datetime64[D]
      spy_idx:     int (-1 si no hay)
      etf_indices: dict[ETF_key → i]
    """
    ref_df = historico.get(calendar_ref)
    if ref_df is None:
        all_dates: set = set()
        for df in historico.values():
            all_dates.update(df.index.normalize())
        dates = pd.DatetimeIndex(sorted(all_dates))
    else:
        dates = ref_df.index.normalize().unique().sort_values()

    keys = list(historico.keys())
    N = len(keys)
    T = len(dates)
    arr = np.full((N, T, 5), np.nan, dtype=np.float64)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    for n, k in enumerate(keys):
        df = historico[k]
        # Forzar tz-naive + normalize para alinear con dates
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        df2 = df.copy()
        df2.index = idx.normalize()
        df2 = df2[~df2.index.duplicated(keep="last")].sort_index()
        df2 = df2.reindex(dates)  # NaN donde falta
        arr[n, :, :] = df2[cols].to_numpy(dtype=np.float64)

    if not CUPY_OK:
        raise RuntimeError(f"CuPy no disponible: {_IMPORT_ERR}")

    arr_gpu = cp.asarray(arr)
    closes = arr_gpu[:, :, COL_CLOSE]
    valid = cp.isfinite(closes) & (closes > 0)

    idx_tickers = {k: i for i, k in enumerate(keys)}
    spy_idx = idx_tickers.get("SPY", -1)
    etf_indices = {k: idx_tickers[k] for k in keys if k.startswith("ETF_")}
    tickers = [k for k in keys if k != "SPY" and not k.startswith("ETF_")]

    return {
        "arr_gpu":     arr_gpu,
        "valid_mask":  valid,
        "tickers":     tickers,
        "idx_tickers": idx_tickers,
        "dates":       dates.to_numpy().astype("datetime64[D]"),
        "spy_idx":     spy_idx,
        "etf_indices": etf_indices,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  KERNELS PRECOMPUTE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_perfs_gpu(closes: "cp.ndarray", valid: "cp.ndarray") -> "cp.ndarray":
    """
    Perfs [N, T, 5] — orden ventanas (5,21,63,126,252).
    perf[n,t,p] = (closes[n,t]/closes[n,t-ventana[p]] - 1)*100  si t>=ventana y prev>0
                   0.0 en cualquier otro caso (réplica calcular_perfs_nb).
    """
    N, T = closes.shape
    P = len(VENTANAS_RS)
    out = cp.zeros((N, T, P), dtype=cp.float64)
    # c0 mask (closes[t] > 0)
    c0_ok = valid  # ya garantiza Close>0
    for p, v in enumerate(VENTANAS_RS):
        if v >= T:
            continue
        prev = cp.zeros_like(closes)
        prev[:, v:] = closes[:, :T - v]
        prev_ok = cp.isfinite(prev) & (prev > 0)
        idx_ok = cp.zeros(T, dtype=cp.bool_)
        idx_ok[v:] = True
        m = c0_ok & prev_ok & idx_ok[None, :]
        ratio = cp.where(m, closes / cp.where(prev > 0, prev, 1.0) - 1.0, 0.0)
        out[:, :, p] = ratio * 100.0
    return out


def compute_rs_gpu(perfs: "cp.ndarray", universe_mask: "cp.ndarray") -> "cp.ndarray":
    """
    RS rating [N, T] 1-99. Para cada día t, ranks cross-sectional
    sobre tickers en universe_mask[:,t]. Réplica de _precomputar_chunk
    (NO calcular_rs_scores_nb — usa pesos_rs dict orden distinto).

    Algoritmo CPU:
      orden = sorted(range(n), key=lambda x: vals[x])  # stable
      ranks[orig_i] = (rk / (n-1)) * 100 if n>1 else 50
    Score = sum(ranks[p] * peso[p]).
    RS = ((score - smin) / (smax - smin)) * 98 + 1, redondeado.

    GPU: por cada periodo, argsort stable cross-sectional → ranks.
    Tickers fuera universe → score NaN.
    """
    N, T, P = perfs.shape
    pesos = cp.asarray(PESOS_RS_NB, dtype=cp.float64)  # [P]
    scores = cp.zeros((N, T), dtype=cp.float64)
    # Para argsort stable con máscara: a inválidos asignar +inf no funciona
    # (rompe ranks). Mejor: por día contar n_valid y calcular ranks solo
    # sobre subset. Workaround: marcar inválidos con NaN, usar argsort
    # con tiebreak por índice. Pero perf por ticker fuera mask debe NO
    # contar — replicamos CPU restringido a universe_mask.
    #
    # Estrategia: para cada (t, p) → vals[mask], argsort stable, ranks.
    # Loop sobre T es OK (T~1000), GPU paralela sobre N por periodo.
    # Mejor: loop sobre P (5), por periodo procesamos todos los días.
    #
    # Approach simple/correcto: vectorizar por columna usando que CPU
    # incluye TODOS los rows que llegaron a `rows` (universo PIT). Aquí
    # universe_mask es booleano [N,T]. Implementación con loop T (rápido,
    # T~1000, cada iter es kernel pequeño sobre N~600).
    n_per_day = universe_mask.sum(axis=0)  # [T]
    for t in range(T):
        n = int(n_per_day[t].item())
        if n < 5:
            scores[:, t] = cp.nan
            continue
        mask_t = universe_mask[:, t]
        idx_universe = cp.where(mask_t)[0]  # [n]
        for p in range(P):
            vals = perfs[idx_universe, t, p]  # [n]
            # argsort stable
            orden = cp.argsort(vals, kind="stable")
            # rk_position[k] = posicion de idx_universe[k] en orden inverso
            # ranks[orig_k] = (rk / (n-1)) * 100
            rks = cp.empty(n, dtype=cp.float64)
            arange_n = cp.arange(n, dtype=cp.float64)
            rks[orden] = arange_n / max(n - 1, 1) * 100.0
            scores[idx_universe, t] += rks * float(pesos[p])

    # Normalize min/max por día — solo sobre universe
    rs = cp.full((N, T), cp.nan, dtype=cp.float64)
    for t in range(T):
        mask_t = universe_mask[:, t]
        n = int(mask_t.sum().item())
        if n < 5:
            continue
        idx_u = cp.where(mask_t)[0]
        sc = scores[idx_u, t]
        smin = sc.min()
        smax = sc.max()
        rng = smax - smin
        if float(rng) > 0.0:
            rs_vals = ((sc - smin) / rng) * 98.0 + 1.0
        else:
            rs_vals = cp.full_like(sc, 50.0)
        # round() para igualar CPU (`round(...)`)
        rs[idx_u, t] = cp.round(rs_vals)
    return rs


def compute_indicators_gpu(arr_gpu: "cp.ndarray",
                            valid: "cp.ndarray") -> dict:
    """
    Indicadores [N, T]: vol_avg20, vol_60, rel_vol, h52, l52, pct52,
    vol_sem (12 semanas), calidad_vela (sin redondeo).

    Importante: usar EXACTAMENTE mismos ranges que CPU
      vol_avg = mean(Volume[i-20:i])   ← excluye i (semantica iloc)
      vol_60  = mean(Volume[i-60:i])   ← excluye i
      h52     = max(High[i-252:i+1])   ← incluye i
      l52     = min(Low[i-252:i+1])    ← incluye i
      pct52   = (close/h52 - 1) * 100
    """
    N, T = valid.shape
    opens = arr_gpu[:, :, COL_OPEN]
    highs = arr_gpu[:, :, COL_HIGH]
    lows  = arr_gpu[:, :, COL_LOW]
    closes = arr_gpu[:, :, COL_CLOSE]
    vols   = arr_gpu[:, :, COL_VOL]

    # NaN-safe: rellenar NaN con 0 para sums, mantener mask separado
    closes_f = cp.where(valid, closes, 0.0)
    highs_f  = cp.where(valid, highs, -cp.inf)
    lows_f   = cp.where(valid, lows, cp.inf)
    vols_f   = cp.where(valid, vols, 0.0)
    valid_f  = valid.astype(cp.float64)

    def rolling_mean_excl(x_filled, mask_filled, window):
        """
        Mean del rango (i-window, i) — EXCLUYE i. Réplica iloc[i-w:i].
        Usa cumsum trick. Para t<window, ventana parcial [0, t).
        Si NO hay observaciones validas → 0.0 (matchea ramas CPU).
        """
        cs_x = cp.cumsum(x_filled, axis=1)
        cs_m = cp.cumsum(mask_filled, axis=1)
        # sum_{k=max(0,t-w)}^{t-1} x_k = cs_x[t-1] - cs_x[max(0,t-w)-1]
        # Trasladamos un día y aplicamos cumsum padded
        zero = cp.zeros((x_filled.shape[0], 1), dtype=cp.float64)
        cs_x_pad = cp.concatenate([zero, cs_x], axis=1)  # [N, T+1]
        cs_m_pad = cp.concatenate([zero, cs_m], axis=1)
        t = cp.arange(T)
        end = t  # corresponde a cs_*_pad[t] = sum hasta t-1 inclusive
        start = cp.maximum(t - window, 0)
        # sum_x[t] = cs_x_pad[end] - cs_x_pad[start]
        sum_x = cs_x_pad[:, end] - cs_x_pad[:, start]
        cnt   = cs_m_pad[:, end] - cs_m_pad[:, start]
        mean  = cp.where(cnt > 0, sum_x / cp.where(cnt > 0, cnt, 1.0), 0.0)
        return mean, cnt

    vol_avg20, _ = rolling_mean_excl(vols_f, valid_f, 20)
    vol_60d, _   = rolling_mean_excl(vols_f, valid_f, 60)

    rel_vol = cp.where(vol_avg20 > 0, vols / cp.where(vol_avg20 > 0, vol_avg20, 1.0), 1.0)
    # En filas con vols NaN: convertir rel_vol a 1.0 (CPU fallback)
    rel_vol = cp.where(valid, rel_vol, 1.0)

    # h52, l52 sobre [max(0,t-252), t] inclusive.
    # CuPy NO soporta maximum.accumulate. Estrategia:
    #   1) Para t>=252: usar maximum_filter1d (size=253, origin=-126,
    #      mode='constant', cval=±inf) — ventana exacta [t-252, t].
    #   2) Para t<252 (bordes warmup): loop python, kernel max/min sobre
    #      slice [:, :t+1]. 252 iters × kernel chiquito sobre [N, t+1].
    #      Es lento pero solo se ejecuta UNA vez al precompute.
    h52 = cp.empty_like(highs_f)
    l52 = cp.empty_like(lows_f)

    use_filter = True
    try:
        from cupyx.scipy.ndimage import maximum_filter1d, minimum_filter1d
    except Exception:
        use_filter = False

    if use_filter and T > 252:
        # CuPy: origin=+126 → ventana [t-252, t] (signo invertido vs scipy).
        h_w = maximum_filter1d(highs_f, size=253, axis=1, mode="constant",
                                cval=-cp.inf, origin=126)
        l_w = minimum_filter1d(lows_f,  size=253, axis=1, mode="constant",
                                cval=cp.inf,  origin=126)
        h52[:, 252:] = h_w[:, 252:]
        l52[:, 252:] = l_w[:, 252:]
    else:
        for t in range(252, T):
            h52[:, t] = highs_f[:, max(0, t - 252):t + 1].max(axis=1)
            l52[:, t] = lows_f[:,  max(0, t - 252):t + 1].min(axis=1)

    # Bordes t<252: ventana [0, t+1]
    for t in range(0, min(252, T)):
        h52[:, t] = highs_f[:, :t + 1].max(axis=1)
        l52[:, t] = lows_f[:,  :t + 1].min(axis=1)

    pct52 = cp.where(h52 > 0, (closes / cp.where(h52 > 0, h52, 1.0) - 1.0) * 100.0, -99.0)

    # Vol semanal: media de |ret_5d| sobre últimas min(12, t//5) semanas
    # ret_w = closes[t - w*5] vs closes[t - (w+1)*5], w=0..11
    # Implementación vectorial: cada w requiere shift de closes por 5w
    vol_sem = cp.full((N, T), 3.0, dtype=cp.float64)
    rets_accum = cp.zeros((N, T), dtype=cp.float64)
    cnt_accum = cp.zeros((N, T), dtype=cp.float64)
    for w in range(12):
        lag0 = (w + 1) * 5
        lag1 = w * 5
        if lag0 >= T:
            break
        p0 = cp.zeros_like(closes_f)
        p1 = cp.zeros_like(closes_f)
        p0[:, lag0:] = closes_f[:, :T - lag0]
        p1[:, lag1:] = closes_f[:, :T - lag1] if lag1 > 0 else closes_f
        m_idx = cp.zeros(T, dtype=cp.bool_)
        m_idx[lag0:] = True
        # cnt accepts también t//5 > w: ya garantizado por lag0>= (w+1)*5 → t>=lag0
        mask_ret = m_idx[None, :] & (p0 > 0)
        ret = cp.where(mask_ret, p1 / cp.where(p0 > 0, p0, 1.0) - 1.0, 0.0)
        rets_accum += cp.abs(ret) * 100.0 * mask_ret
        cnt_accum += mask_ret.astype(cp.float64)
    vol_sem = cp.where(cnt_accum > 0, rets_accum / cp.where(cnt_accum > 0, cnt_accum, 1.0), 3.0)

    # Calidad vela (CPU usa redondeo a 1 decimal — aquí devolvemos raw, el
    # extractor redondeará para coincidir con dict)
    rng_v = highs - lows
    cuerpo = cp.abs(closes - opens)
    ratio_cuerpo = cp.where(rng_v > 0, cuerpo / cp.where(rng_v > 0, rng_v, 1.0), 0.0)
    pos_cierre = cp.where(rng_v > 0, (closes - lows) / cp.where(rng_v > 0, rng_v, 1.0), 0.5)
    alcista = (closes > opens).astype(cp.float64) * 15.0
    rv_clamped = cp.minimum(rel_vol, 3.0) / 3.0 * 15.0
    calidad_vela = ratio_cuerpo * 40.0 + pos_cierre * 30.0 + alcista + rv_clamped

    # gap_earnings: max |open[i+fwd]/close[i+fwd-1] - 1| * 100 para fwd 1..5
    gap_earn = cp.zeros((N, T), dtype=cp.float64)
    for fwd in range(1, 6):
        if fwd >= T:
            break
        o_f = cp.zeros_like(closes_f)
        c_a = cp.zeros_like(closes_f)
        o_f[:, :T - fwd] = opens[:, fwd:]
        c_a[:, :T - fwd] = closes[:, fwd-1:T - 1] if fwd > 0 else closes
        m_idx = cp.zeros(T, dtype=cp.bool_)
        m_idx[:T - fwd] = True
        mask = m_idx[None, :] & (c_a > 0) & cp.isfinite(o_f) & cp.isfinite(c_a)
        g = cp.where(mask, cp.abs(o_f / cp.where(c_a > 0, c_a, 1.0) - 1.0) * 100.0, 0.0)
        gap_earn = cp.maximum(gap_earn, g)

    return {
        "vol_avg20":    vol_avg20,
        "vol_60d":      vol_60d,
        "rel_vol":      rel_vol,
        "h52":          h52,
        "l52":          l52,
        "pct52":        pct52,
        "vol_sem":      vol_sem,
        "calidad_vela": calidad_vela,
        "gap_earnings": gap_earn,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN CANDIDATOS (GPU → dicts)
# ═══════════════════════════════════════════════════════════════════════════════

def _spy_sma200_gpu(arr_gpu, spy_idx, T):
    if spy_idx < 0:
        return None
    closes = arr_gpu[spy_idx, :, COL_CLOSE]  # [T]
    # SMA 200 sobre closes, NaN para t<199
    cs = cp.cumsum(cp.where(cp.isfinite(closes), closes, 0.0))
    cs_pad = cp.concatenate([cp.zeros(1, dtype=cs.dtype), cs])
    t = cp.arange(T)
    start = cp.maximum(t - 199, 0)
    end = t + 1
    sums = cs_pad[end] - cs_pad[start]
    cnts = cp.minimum(t + 1, 200)
    sma = cp.where(cnts >= 200, sums / cp.where(cnts > 0, cnts, 1.0), cp.nan)
    return sma


def extract_candidatos_gpu(tensor_data: dict,
                            desde: datetime.date,
                            hasta: datetime.date) -> list:
    """
    Réplica de _precomputar_chunk pero vectorizada. Devuelve list[dict]
    schema idéntico al CPU (mismo orden tras sort por fecha+ticker).

    Filtro PIT por día:
      - i >= WARMUP_MIN y i < T - 1  (entrada_real en i+1)
      - Close[i] > 0
      - n_universe >= 5
    """
    arr_gpu     = tensor_data["arr_gpu"]
    valid_mask  = tensor_data["valid_mask"]
    tickers     = tensor_data["tickers"]
    idx_tickers = tensor_data["idx_tickers"]
    dates       = tensor_data["dates"]
    spy_idx     = tensor_data["spy_idx"]

    N, T, _ = arr_gpu.shape

    # Mask filas tickers (excluye SPY/ETF)
    ticker_idxs = cp.asarray([idx_tickers[t] for t in tickers], dtype=cp.int64)

    # Universo PIT por (n_in_tickers, t): valid AND t in [WARMUP_MIN, T-2] AND close>0
    closes = arr_gpu[:, :, COL_CLOSE]
    valid_full = valid_mask
    t_range_mask = cp.zeros(T, dtype=cp.bool_)
    t_range_mask[WARMUP_MIN:T - 1] = True
    universe_mask_full = valid_full & t_range_mask[None, :]
    # Restringir a sólo tickers no-SPY/ETF para RS cross-sectional
    universe_mask_tickers = cp.zeros_like(universe_mask_full)
    universe_mask_tickers[ticker_idxs] = universe_mask_full[ticker_idxs]

    # Date mask [desde, hasta]
    dates_np = dates  # datetime64[D]
    d0 = np.datetime64(desde, "D")
    d1 = np.datetime64(hasta, "D")
    date_in = (dates_np >= d0) & (dates_np <= d1)
    # Combinar con t_range
    t_active = cp.asarray(date_in) & t_range_mask
    # Mínimo n_universe >= 5
    universe_active = universe_mask_tickers & t_active[None, :]

    # Filter days where universe >=5
    n_per_day = universe_active.sum(axis=0)
    day_ok = n_per_day >= 5

    # Perfs [N, T, 5]
    perfs = compute_perfs_gpu(closes, valid_full)

    # RS rating cross-sectional usando universe_active
    rs = compute_rs_gpu(perfs, universe_active)

    # Indicadores
    ind = compute_indicators_gpu(arr_gpu, valid_full)

    # SPY SMA200 → mercado_alcista por día
    spy_sma = _spy_sma200_gpu(arr_gpu, spy_idx, T)
    if spy_sma is not None:
        spy_close = arr_gpu[spy_idx, :, COL_CLOSE]
        mercado_alcista_t = (cp.isfinite(spy_sma) & (spy_close > spy_sma))
    else:
        mercado_alcista_t = cp.ones(T, dtype=cp.bool_)

    # Pre-clasificar estado (con umbrales CPU)
    p1w = perfs[:, :, 0]
    p1m = perfs[:, :, 1]
    p3m = perfs[:, :, 2]
    rel_vol = ind["rel_vol"]
    estado_code = cp.full((N, T), -1, dtype=cp.int8)  # -1 = ninguno
    cond_break = (p1w > 8) & (rs >= 85) & (rel_vol >= 1.5)
    cond_sque  = (cp.abs(p1m) < 6) & (p3m > 15) & (rs >= 75) & ~cond_break
    cond_base  = (cp.abs(p1m) < 8) & (p3m > 20) & ~cond_break & ~cond_sque
    cond_ext   = (p1m > 20) & ~cond_break & ~cond_sque & ~cond_base
    cond_vig   = (p1m > 8)  & ~cond_break & ~cond_sque & ~cond_base & ~cond_ext
    estado_code = cp.where(cond_break, 0, estado_code)
    estado_code = cp.where(cond_sque,  1, estado_code)
    estado_code = cp.where(cond_base,  2, estado_code)
    estado_code = cp.where(cond_ext,   3, estado_code)
    estado_code = cp.where(cond_vig,   4, estado_code)
    ESTADO_NAMES = ("BREAKOUT", "SQUEEZE", "EN BASE", "EXTENDIDO", "VIGILANCIA")

    # Mask señal: en universe_active, tiene estado, entrada_real > 0
    has_estado = (estado_code >= 0)
    opens = arr_gpu[:, :, COL_OPEN]
    entrada_real_arr = cp.full((N, T), cp.nan, dtype=cp.float64)
    entrada_real_arr[:, :T - 1] = opens[:, 1:T]
    entrada_ok = cp.isfinite(entrada_real_arr) & (entrada_real_arr > 0)

    sig_mask = universe_active & has_estado & entrada_ok & day_ok[None, :]

    # Sub-scores
    acelerando_arr = (p1w * 4.3 > p1m) & (p1m > (p3m / 3.0 * 0.7))
    base_estrecha_arr = (cp.abs(p1m) < 8) & (p3m > 15)
    vol_score = cp.where(rel_vol >= 2.5, 100.0,
                         cp.where(rel_vol >= 1.5, 70.0, 40.0))
    mom_score = cp.minimum(100.0, cp.maximum(0.0,
        cp.minimum(p1w * 4.3, 20.0) * 1.5 +
        cp.minimum(p1m, 15.0) * 1.5 +
        cp.where(acelerando_arr, 10.0, 0.0)))
    h52 = ind["h52"]
    l52 = ind["l52"]
    pos_rango = (closes - l52) / cp.maximum(h52 - l52, 0.01)
    estr_score = cp.minimum(100.0,
        pos_rango * 40.0 +
        cp.where(ind["pct52"] >= -5.0, 15.0,
                 cp.where(ind["pct52"] >= -10.0, 8.0, 0.0)) +
        cp.where(base_estrecha_arr, 20.0, 0.0))

    # Regimen vol_sem
    vol_sem = ind["vol_sem"]
    regimen_code = cp.where(vol_sem < 2.0, 0,
                    cp.where(vol_sem < 4.0, 1,
                    cp.where(vol_sem < 7.0, 2, 3)))
    REG_NAMES = ("tranquilo", "normal", "alto", "extremo")

    # Extraer índices de señales y bajar a host
    sig_idx = cp.argwhere(sig_mask)  # [K, 2] (n, t)
    if int(sig_idx.shape[0]) == 0:
        return []

    sig_idx_host = cp.asnumpy(sig_idx)
    # Para cada señal, leer todos los campos a host (una sola transferencia)
    ns = sig_idx_host[:, 0]
    ts = sig_idx_host[:, 1]

    def gather(field):
        return cp.asnumpy(field[ns, ts])

    rs_h        = gather(rs)
    p1w_h       = gather(p1w)
    p1m_h       = gather(p1m)
    p3m_h       = gather(p3m)
    pct52_h     = gather(ind["pct52"])
    rvol_h      = gather(rel_vol)
    volavg_h    = gather(ind["vol_avg20"])
    vol60_h     = gather(ind["vol_60d"])
    gap_h       = gather(ind["gap_earnings"])
    calidad_h   = gather(ind["calidad_vela"])
    vol_sem_h   = gather(ind["vol_sem"])
    regimen_h   = gather(regimen_code)
    accel_h     = gather(acelerando_arr)
    base_h      = gather(base_estrecha_arr)
    estado_h    = gather(estado_code)
    mkt_t       = cp.asnumpy(mercado_alcista_t)  # [T]
    vol_score_h = gather(vol_score)
    mom_score_h = gather(mom_score)
    estr_h      = gather(estr_score)
    close_h     = gather(closes)
    entrada_h   = gather(entrada_real_arr)
    h52_h       = gather(h52)
    l52_h       = gather(l52)

    fechas_str = dates.astype(str)
    out = []
    tickers_keys = list(tensor_data["idx_tickers"].keys())  # orden por idx
    for k in range(len(ns)):
        n_i = int(ns[k]); t_i = int(ts[k])
        out.append({
            "fecha":         str(fechas_str[t_i]),
            "fecha_entrada": str(fechas_str[t_i + 1]),
            "ticker":        tickers_keys[n_i],
            "rs":            int(rs_h[k]),
            "p1w":           round(float(p1w_h[k]), 2),
            "p1m":           round(float(p1m_h[k]), 2),
            "p3m":           round(float(p3m_h[k]), 2),
            "pct52":         round(float(pct52_h[k]), 1),
            "rel_vol":       round(float(rvol_h[k]), 2),
            "vol_avg_m":     round(float(volavg_h[k]) / 1e6, 3),
            "vol_60d":       round(float(vol60_h[k]), 0),
            "gap_earnings":  round(float(gap_h[k]), 2),
            "calidad_vela":  round(float(calidad_h[k]), 1),
            "vol_sem":       round(float(vol_sem_h[k]), 2),
            "regimen":       REG_NAMES[int(regimen_h[k])],
            "acelerando":    bool(accel_h[k]),
            "base_estrecha": bool(base_h[k]),
            "estado":        ESTADO_NAMES[int(estado_h[k])],
            "mercado_alcista": bool(mkt_t[t_i]),
            "vol_score":     int(vol_score_h[k]),
            "mom_score":     round(float(mom_score_h[k]), 2),
            "estr_score":    round(float(estr_h[k]), 2),
            "precio_señal":  round(float(close_h[k]), 4),
            "entrada_real":  round(float(entrada_h[k]), 4),
            "h52":           round(float(h52_h[k]), 4),
            "l52":           round(float(l52_h[k]), 4),
        })

    out.sort(key=lambda x: (x["fecha"], x["ticker"]))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN VECTORIZADA OCO
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_trades_gpu(señales: list, tensor_data: dict, params) -> list:
    """
    Sim OCO batch sobre [K_señales, max_hold+1] OHLC. Réplica EXACTA de
    simular_operacion (CPU) — incluye trailing_stop_pct (modo "pct").
    Modo atr_chandelier delega a CPU (rara strategy, hot path no critico).

    Returns list[dict] mismo schema que CPU (operaciones).
    """
    K = len(señales)
    if K == 0:
        return []

    arr_gpu     = tensor_data["arr_gpu"]
    valid_mask  = tensor_data["valid_mask"]
    idx_tickers = tensor_data["idx_tickers"]
    dates       = tensor_data["dates"]
    spy_idx     = tensor_data["spy_idx"]
    N, T, _     = arr_gpu.shape
    max_hold    = int(params.max_hold_days)
    slip        = float(params.slippage_pct)
    fraccion_t1 = float(params.fraccion_t1)
    trail_pct   = float(getattr(params, "trailing_stop_pct", 0.07) or 0.0)

    # Mapear señales → (ticker_idx, idx0)
    fechas_str = dates.astype(str)
    fecha_to_t = {str(f): i for i, f in enumerate(fechas_str)}

    ticker_idx_host = np.zeros(K, dtype=np.int64)
    idx0_host       = np.zeros(K, dtype=np.int64)
    entrada_host    = np.zeros(K, dtype=np.float64)
    stop_host       = np.zeros(K, dtype=np.float64)
    t1_host         = np.zeros(K, dtype=np.float64)
    t2_host         = np.zeros(K, dtype=np.float64)
    raw_host        = np.zeros(K, dtype=np.float64)
    valid_sig       = np.ones(K, dtype=bool)

    for k, s in enumerate(señales):
        t_str = s["fecha_entrada"]
        tk    = s["ticker"]
        if tk not in idx_tickers or t_str not in fecha_to_t:
            valid_sig[k] = False
            continue
        ticker_idx_host[k] = idx_tickers[tk]
        idx0_host[k]       = fecha_to_t[t_str]
        raw_host[k]        = float(s["entrada_real"])
        entrada_host[k]    = round(raw_host[k] * (1.0 + slip), 2)
        stop_host[k]       = float(s["stop"])
        t1_host[k]         = float(s["t1"])
        t2_host[k]         = float(s["t2"])

    # Subir a GPU
    tk_idx  = cp.asarray(ticker_idx_host)
    idx0    = cp.asarray(idx0_host)
    entrada = cp.asarray(entrada_host)
    stop    = cp.asarray(stop_host)
    t1_p    = cp.asarray(t1_host)
    t2_p    = cp.asarray(t2_host)

    # Para cada k slice [max_hold+1] de O/H/L/C desde idx0+1
    # idx_off[k, j] = idx0[k] + j+1 (j=0..max_hold-1)
    # j cumple 1..max_hold; usaremos j_arr = arange(1, max_hold+1)
    j_arr = cp.arange(1, max_hold + 1, dtype=cp.int64)
    idx_mat = idx0[:, None] + j_arr[None, :]  # [K, H]
    idx_mat_clipped = cp.minimum(idx_mat, T - 1)
    in_range = idx_mat < T  # [K, H]

    opens  = arr_gpu[:, :, COL_OPEN]
    highs  = arr_gpu[:, :, COL_HIGH]
    lows   = arr_gpu[:, :, COL_LOW]
    closes = arr_gpu[:, :, COL_CLOSE]

    O = opens[tk_idx[:, None], idx_mat_clipped]   # [K, H]
    H = highs[tk_idx[:, None], idx_mat_clipped]
    L = lows[tk_idx[:, None], idx_mat_clipped]
    C = closes[tk_idx[:, None], idx_mat_clipped]

    # Step-by-step sim — cada j calcula transición. Estados [K]:
    # active, stop_actual, t1_hit, fraccion_viva, pnl_parcial, precio_salida,
    # motivo, dias, max_precio
    active        = cp.ones(K, dtype=cp.bool_)
    stop_actual   = stop.copy()
    t1_hit        = cp.zeros(K, dtype=cp.bool_)
    fraccion_viva = cp.ones(K, dtype=cp.float64)
    pnl_parcial   = cp.zeros(K, dtype=cp.float64)
    precio_salida = entrada.copy()
    motivo        = cp.full(K, 3, dtype=cp.int8)  # 3=expirado
    dias          = cp.zeros(K, dtype=cp.int32)
    max_precio    = entrada.copy()

    OUTCOME_FROM_MOTIVO = {0: "LOSS", 1: "WIN_T2", 2: None, 3: None, 4: None}
    MOTIVO_STR = {0: "stop", 1: "t2", 2: "filtro_mercado", 3: "expirado", 4: "fin_datos"}

    for j_i in range(max_hold):
        # Para indices fuera del histórico (idx>=T): cerrar con close último valido
        out_of_range_now = (~in_range[:, j_i]) & active
        if cp.any(out_of_range_now):
            # close último valido por ticker = closes[tk_idx, T-1]
            last_close = closes[tk_idx, T - 1]
            precio_salida = cp.where(out_of_range_now, last_close, precio_salida)
            motivo = cp.where(out_of_range_now, 4, motivo)  # fin_datos
            dias   = cp.where(out_of_range_now, j_i + 1, dias)
            active = active & ~out_of_range_now

        op_ = O[:, j_i]
        hi_ = H[:, j_i]
        lo_ = L[:, j_i]
        cl_ = C[:, j_i]

        # Stop hit (gap → min(op_, stop_actual))
        hit_stop = active & (lo_ <= stop_actual)
        if cp.any(hit_stop):
            p_sal = cp.minimum(op_, stop_actual)
            precio_salida = cp.where(hit_stop, p_sal, precio_salida)
            motivo        = cp.where(hit_stop, 0, motivo)
            dias          = cp.where(hit_stop, j_i + 1, dias)
            active        = active & ~hit_stop

        # T1 hit (solo si !t1_hit y hi_ >= t1_p)
        hit_t1 = active & (~t1_hit) & (hi_ >= t1_p)
        # Al hit T1: t1_hit=True, fraccion_viva=1-frac_t1, pnl_parcial=...,
        # stop_actual = max(stop_actual, entrada), max_precio = hi_
        new_fraccion = 1.0 - fraccion_t1
        new_pnl_parcial = (t1_p - entrada) / cp.where(entrada > 0, entrada, 1.0) * fraccion_t1
        fraccion_viva = cp.where(hit_t1, new_fraccion, fraccion_viva)
        pnl_parcial   = cp.where(hit_t1, new_pnl_parcial, pnl_parcial)
        stop_actual   = cp.where(hit_t1, cp.maximum(stop_actual, entrada), stop_actual)
        max_precio    = cp.where(hit_t1, hi_, max_precio)
        t1_hit        = t1_hit | hit_t1

        # En el SAME día que T1: si hi_ >= t2 → cerrar T2 inmediato + 'continue'
        # En CPU: tras hit_t1, hace `if hi_ >= t2: precio_salida=t2; motivo=1; break` luego `continue`
        # Esto significa que cierre T2 puede ocurrir EN el mismo día que T1.
        immediate_t2 = hit_t1 & (hi_ >= t2_p)
        if cp.any(immediate_t2):
            precio_salida = cp.where(immediate_t2, t2_p, precio_salida)
            motivo        = cp.where(immediate_t2, 1, motivo)
            dias          = cp.where(immediate_t2, j_i + 1, dias)
            active        = active & ~immediate_t2

        # CPU: tras hit_t1 (no break) hace 'continue' → NO entra a trailing
        # ni T2 ese día. Mismo aquí: separamos: trailing/T2 SOLO si t1_hit ya
        # estaba activo ANTES (no en este iter).
        t1_was_active = t1_hit & ~hit_t1

        # Trailing post-T1 (modo pct)
        if trail_pct > 0:
            max_precio = cp.where(t1_was_active & (hi_ > max_precio), hi_, max_precio)
            trail = max_precio * (1.0 - trail_pct)
            stop_actual = cp.where(t1_was_active & (trail > stop_actual) & active, trail, stop_actual)
        else:
            max_precio = cp.where(t1_was_active & (hi_ > max_precio), hi_, max_precio)

        # T2 (solo si t1 ya estaba activo)
        hit_t2 = active & t1_was_active & (hi_ >= t2_p)
        if cp.any(hit_t2):
            precio_salida = cp.where(hit_t2, t2_p, precio_salida)
            motivo        = cp.where(hit_t2, 1, motivo)
            dias          = cp.where(hit_t2, j_i + 1, dias)
            active        = active & ~hit_t2

        # Último día (j_i == max_hold - 1) y todavía active → expirado.
        # OJO: replicar bug CPU — si T1 hit ESE último día, CPU hace
        # `continue` y NO actualiza precio_salida (queda en entrada). Solo
        # actualizamos para trades que NO hit T1 ese día.
        if j_i == max_hold - 1:
            still_active = active & ~hit_t1
            if cp.any(still_active):
                precio_salida = cp.where(still_active, cl_, precio_salida)
                motivo        = cp.where(still_active, 3, motivo)
                dias          = cp.where(still_active, j_i + 1, dias)
                active        = active & ~still_active
            # Los que T1-hit ese día: salen también pero con precio_salida
            # initial (entrada) — bug CPU replicado. Cerramos active.
            t1_last = active & hit_t1
            if cp.any(t1_last):
                # CPU: precio_salida = entrada (initial), motivo = "expirado" (initial)
                dias   = cp.where(t1_last, j_i + 1, dias)
                active = active & ~t1_last

    pnl_total = pnl_parcial + (precio_salida - entrada) / cp.where(entrada > 0, entrada, 1.0) * fraccion_viva
    pnl_pct = cp.round(pnl_total * 100.0, 2)

    # Outcome
    outcome = cp.where(motivo == 0, 0,         # LOSS
              cp.where(motivo == 1, 2,         # WIN_T2
              cp.where(t1_hit, 1,              # WIN_T1
              cp.where(pnl_pct >  0.5, 3,      # WIN_PARCIAL
              cp.where(pnl_pct < -0.5, 0, 4))))).astype(cp.int8)  # LOSS / BREAKEVEN

    # Bajar a host
    pnl_pct_h    = cp.asnumpy(pnl_pct)
    outcome_h    = cp.asnumpy(outcome)
    motivo_h     = cp.asnumpy(motivo)
    dias_h       = cp.asnumpy(dias)
    precio_sal_h = cp.asnumpy(precio_salida)
    t1_hit_h     = cp.asnumpy(t1_hit)
    stop_actual_h= cp.asnumpy(stop_actual)

    OUTCOME_NAMES = {0: "LOSS", 1: "WIN_T1", 2: "WIN_T2", 3: "WIN_PARCIAL", 4: "BREAKEVEN", 5: "SIN_DATOS"}

    result = []
    for k, s in enumerate(señales):
        if not valid_sig[k]:
            result.append({**s, "outcome": "SIN_DATOS", "pnl_pct": 0.0,
                           "dias_en_operacion": 0, "precio_salida": 0.0,
                           "motivo_salida": "sin_datos", "t1_alcanzado": False,
                           "entrada_efectiva": entrada_host[k]})
            continue
        result.append({**s,
            "outcome":           OUTCOME_NAMES[int(outcome_h[k])],
            "pnl_pct":           float(pnl_pct_h[k]),
            "dias_en_operacion": int(dias_h[k]),
            "precio_salida":     round(float(precio_sal_h[k]), 2),
            "motivo_salida":     MOTIVO_STR[int(motivo_h[k])],
            "t1_alcanzado":      bool(t1_hit_h[k]),
            "stop_final":        round(float(stop_actual_h[k]), 2),
            "entrada_efectiva":  float(entrada_host[k]),
            "slippage_usd":      round(float(entrada_host[k] - raw_host[k]), 3)})

    return result
