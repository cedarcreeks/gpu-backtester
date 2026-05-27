"""
backtest_core.py — Motor de Backtesting Puro, fiel a analyzer.py
═══════════════════════════════════════════════════════════════════════════════
Funciones puras sin globals — diseñado para ser llamado en paralelo desde
backtest_wf.py (walk-forward) con distintas combinaciones de parámetros.

DIFERENCIA CLAVE respecto a la versión anterior:
  El cálculo de RS Rating usa los 5 períodos y pesos EXACTOS de analyzer.py:
    1Y=20%, 6M=20%, 3M=25%, 1M=25%, 1W=10%
  (La versión anterior usaba solo 1W×40% + 1M×20% + 3M×40%)

  clasificar_estado() replica EXACTAMENTE los umbrales de analyzer.py.
  calcular_posicion replica EXACTAMENTE la lógica de entrada y stop.

Interfaz pública:
  Params                    — dataclass con todos los parámetros
  run_backtest()            — punto de entrada: historico + params → métricas
  run_backtest_rapido()     — versión rápida con candidatos pre-computados
  precomputar_candidatos()  — escanea el histórico una vez por ventana
  filtrar_candidatos()      — aplica Params sobre candidatos (sin DataFrames)
  construir_señales()       — escaneo paralelo completo
  simular_operacion()       — ciclo de vida de una operación
  simular_portfolio()       — gestión de portfolio con límite de posiciones
  serializar_historico()    — preparar historico para multiprocessing
  historico_to_numpy()      — compresión numpy para transmisión eficiente
  numpy_to_historico()      — reconstrucción desde numpy
═══════════════════════════════════════════════════════════════════════════════
"""

import datetime
import statistics
import multiprocessing
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError("pip install pandas numpy yfinance")

# Aceleración Numba + caché Parquet (bt_cache.py en la misma carpeta)
try:
    from gpu_bt.cache import (calcular_perfs_nb, calcular_rs_scores_nb,
                          calcular_vol_rel_nb, calcular_vol_avg_nb,
                          calcular_h52_l52_nb, calcular_vol_sem_nb,
                          calcular_calidad_vela_nb, simular_operacion_nb,
                          preparar_arrays_nb, get_idx_nb,
                          _PESOS_NB, OUTCOME_NAMES, MOTIVO_NAMES,
                          descargar_o_cargar as _cache_descargar,
                          warmup_numba, tiene_numba)
    _CACHE_OK = True
except ImportError:
    _CACHE_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Params:
    """
    Todos los parámetros del sistema en un objeto serializable.
    Valores por defecto calibrados con analyzer.py como referencia.
    """
    # ── Filtros de señal (idénticos a settings.py) ────────────────────────────
    rs_minimo:          int   = 50        # settings.RS_MINIMO
    vol_minimo_m:       float = 0.5       # AvgVol mínimo en millones (settings.VOL_MINIMO / 1e6)
    pct_max_52w:        float = -15.0     # settings.PCT_MAX_52W
    score_minimo:       float = 0.0       # 0 = sin filtro
    rvol_minimo:        float = 1.5       # volumen relativo mínimo para BREAKOUT

    # ── Ponderaciones RS (idénticas a settings.PESOS_RS) ─────────────────────
    # No editar — deben coincidir con analyzer.py
    # 1Y=20%, 6M=20%, 3M=25%, 1M=25%, 1W=10%
    # (Se usan como constante en el código — no como campo del dataclass)

    # ── Filtros adicionales ───────────────────────────────────────────────────
    excluir_rs_90_95:   bool  = False     # excluir tramo 90-95 si se detecta valle estructural
    penalizar_vol_alta: bool  = False     # exigir score+2 en régimen de volatilidad alto
    score_vol_extra:    float = 2.0       # puntos extra de score exigidos en régimen alto
    calidad_vela_min:   float = 0.0       # 0 = sin filtro de calidad de vela
    filtro_earnings:    bool  = True      # excluir señales con gap >12% en próximos 5 días
    filtro_mercado:     bool  = True      # solo operar si SPY > SMA200
    excluir_tickers:    tuple = ()

    # ── Gestión de operación (idénticos a settings.py) ────────────────────────
    stop_pct:           float = 0.07      # settings.STOP_PCT
    target_rr_1:        float = 2.0       # settings.TARGET_RR_1
    target_rr_2:        float = 3.0       # settings.TARGET_RR_2
    fraccion_t1:        float = 0.5       # mitad se cierra en T1
    trailing_stop_pct:  float = 0.07      # settings.TRAILING_STOP_PCT (7% bajo máximo post-T1)
    max_hold_days:      int   = 20
    spy_sma_periodo:    int   = 200
    slippage_pct:       float = 0.001     # 0.1%

    # ── Portfolio ─────────────────────────────────────────────────────────────
    max_posiciones:     int   = 5         # settings.MAX_POSICIONES
    capital_total:      float = 10_000.0  # USD simulado
    riesgo_por_op:      float = 0.02      # settings.RIESGO_PCT
    # B10: heat cap — fraccion max del capital comprometida en riesgo abierto
    # simultaneo (suma de stops vivos). 1.0 = sin cap (default, comportamiento
    # baseline). 0.10 = max 10% del capital arriesgado a la vez (con
    # riesgo_por_op=0.02 y max_pos=5 ya da heat=10% al fondo; cap=0.06
    # forzaria max 3 pos al mismo tiempo en regimen de full risk).
    heat_cap:           float = 1.0

    # ── A2: Strategy plugin selector ──────────────────────────────────────────
    # "baseline" = lógica original RS-momentum + BREAKOUT/SQUEEZE (analyzer.py
    # equivalente). Plugins (optional) can be registered via a user-side module exposing
    # filter_fn(candidatos, params) + sim_fn(señal, historico, params).
    strategy_id:        str   = "baseline"
    # dict params específicos de strategy (ej {"atr_mult": 3.0, "ma_period": 20})
    strategy_params:    dict = None

    def __post_init__(self):
        if self.strategy_params is None:
            self.strategy_params = {}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["excluir_tickers"] = list(d.get("excluir_tickers", []))
        return d

    def label(self) -> str:
        """Etiqueta compacta para identificar la combinación en logs.

        Si hay strategy_params (ej. adx_min en adx_trend) se añaden como
        sufijo — sin esto, variantes de strategy_params colapsan en el
        dedup de build_param_grid (mismo label). Baseline (sin
        strategy_params) mantiene el label idéntico al previo.
        """
        trail = f"_tr{self.trailing_stop_pct:.0%}" if self.trailing_stop_pct > 0 else ""
        # heat_cap solo en label cuando < 1.0 (cap activo); 1.0 = sin cap =
        # baseline → label idéntico, no rompe dedup ni reportes existentes.
        hc = f"_hc{int(self.heat_cap*100):02d}" if self.heat_cap < 1.0 else ""
        base = (f"rs{self.rs_minimo}_"
                f"sc{self.score_minimo:.0f}_"
                f"rv{self.rvol_minimo:.1f}_"
                f"vela{self.calidad_vela_min:.0f}_"
                f"earn{'X' if self.filtro_earnings else 'O'}_"
                f"mkt{'X' if self.filtro_mercado else 'O'}"
                f"{trail}{hc}")
        sp = self.strategy_params or {}
        if sp:
            suf = "_".join(f"{k}{v}" for k, v in sorted(sp.items()))
            base = f"{base}_{suf}"
        return base


# ═══════════════════════════════════════════════════════════════════════════════
#  SERIALIZACIÓN DE HISTÓRICO
# ═══════════════════════════════════════════════════════════════════════════════

def serializar_historico(historico: dict) -> tuple:
    """Serializa DataFrames para pasar entre procesos."""
    keys = list(historico.keys())
    data = []
    for df in historico.values():
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        data.append({
            "data":  df.to_dict(orient="list"),
            "index": [str(x.date()) for x in idx],
        })
    return keys, data


def _deserializar_historico(keys, data) -> dict:
    """Reconstruye DataFrames desde el formato serializable."""
    historico = {}
    for k, v in zip(keys, data):
        df = pd.DataFrame(v["data"])
        df.index = pd.to_datetime(v["index"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        historico[k] = df
    return historico


def historico_to_numpy(historico: dict) -> dict:
    """Convierte el histórico a arrays numpy float32 compactos."""
    result = {}
    for t, df in historico.items():
        cols = ["Open", "High", "Low", "Close", "Volume"]
        arr  = df[cols].values.astype(np.float32)
        idx  = np.array([str(d.date()) for d in df.index], dtype=object)
        result[t] = {"arr": arr, "idx": idx}
    return result


def numpy_to_historico(np_hist: dict) -> dict:
    """Reconstruye DataFrames desde el formato numpy compacto."""
    result = {}
    for t, v in np_hist.items():
        raw_idx = v["idx"]
        if raw_idx.dtype == object or raw_idx.dtype.kind in ("U", "S"):
            idx = pd.DatetimeIndex(raw_idx)
        else:
            idx = pd.DatetimeIndex(raw_idx.astype("datetime64[ns]"))
        df = pd.DataFrame(
            v["arr"], index=idx,
            columns=["Open", "High", "Low", "Close", "Volume"],
        ).astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        result[t] = df
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES RS (réplica exacta de analyzer.PESOS_RS y ventanas)
# ═══════════════════════════════════════════════════════════════════════════════

_PESOS_RS = {
    "Perf1Y": 0.20,
    "Perf6M": 0.20,
    "Perf3M": 0.25,
    "Perf1M": 0.25,
    "Perf1W": 0.10,
}
_VENTANAS_RS = {
    "Perf1Y": 252,
    "Perf6M": 126,
    "Perf3M":  63,
    "Perf1M":  21,
    "Perf1W":   5,
}
_WARMUP_MIN = 252   # barras mínimas necesarias antes de la fecha de señal


def _spy_sma(spy_df, periodo: int = 200):
    if spy_df is None:
        return None
    return spy_df["Close"].rolling(periodo).mean()


def _perf(df, i, ventana):
    """Retorno % entre iloc[i-ventana] e iloc[i]. None si no hay datos."""
    if i < ventana:
        return None
    c0 = float(df.iloc[i]["Close"])
    cn = float(df.iloc[i - ventana]["Close"])
    return (c0 / cn - 1) * 100 if cn > 0 else None


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER DE ESCANEO — RÉPLICA EXACTA DE analyzer.py
# ═══════════════════════════════════════════════════════════════════════════════

def _escanear_chunk(args: tuple) -> list:
    """
    Worker puro: escanea un chunk de fechas y devuelve señales.
    Usa Numba para cálculos numéricos si bt_cache está disponible.
    """
    (fechas_chunk, historico_keys, historico_data, params_dict) = args

    import statistics as _stats
    import numpy as _np
    p = Params(**{k: v for k, v in params_dict.items()
                  if k in Params.__dataclass_fields__})

    historico   = _deserializar_historico(historico_keys, historico_data)
    spy_df      = historico.get("SPY")
    spy_sma     = _spy_sma(spy_df, p.spy_sma_periodo)
    tickers     = [t for t in historico if t != "SPY" and not t.startswith("ETF_")]

    # ETFs de sector (prefijo ETF_) — para calcular rs_sector igual que analyzer.py
    # Mapa ETF → nombre interno en el histórico
    _SECTOR_ETF_MAP = {
        "Technology":             "ETF_XLK", "Healthcare":             "ETF_XLV",
        "Financial":              "ETF_XLF", "Financial Services":     "ETF_XLF",
        "Consumer Cyclical":      "ETF_XLY", "Consumer Discretionary": "ETF_XLY",
        "Consumer Defensive":     "ETF_XLP", "Industrials":            "ETF_XLI",
        "Energy":                 "ETF_XLE", "Materials":              "ETF_XLB",
        "Basic Materials":        "ETF_XLB", "Real Estate":            "ETF_XLRE",
        "Utilities":              "ETF_XLU", "Communication Services": "ETF_XLC",
    }
    _etf_keys = list(set(_SECTOR_ETF_MAP.values()))

    def _get_sector_rs_fecha(fecha_str: str) -> dict:
        """Calcula RS de sector ETFs para una fecha dada. Devuelve {ETF_key: rs 1-99}."""
        ts = pd.Timestamp(fecha_str)
        raw = {}
        for etf_key in _etf_keys:
            df_etf = historico.get(etf_key)
            if df_etf is None or len(df_etf) < 65:
                continue
            try:
                i = df_etf.index.get_indexer([ts], method="pad")[0]
                if i < 63:
                    continue
                c0  = float(df_etf.iloc[i]["Close"])
                c5  = float(df_etf.iloc[i - 5]["Close"])   if i >= 5  else c0
                c21 = float(df_etf.iloc[i - 21]["Close"])  if i >= 21 else c0
                c63 = float(df_etf.iloc[i - 63]["Close"])  if i >= 63 else c0
                p1w_e = (c0 / c5  - 1) * 100 if c5  > 0 else 0
                p1m_e = (c0 / c21 - 1) * 100 if c21 > 0 else 0
                p3m_e = (c0 / c63 - 1) * 100 if c63 > 0 else 0
                raw[etf_key] = p1w_e * 0.4 + p1m_e * 0.2 + p3m_e * 0.4
            except Exception:
                continue
        if not raw:
            return {k: 50 for k in _etf_keys}
        vals = list(raw.values())
        v_min, v_max = min(vals), max(vals)
        result = {}
        for k, v in raw.items():
            result[k] = round(((v - v_min) / (v_max - v_min)) * 98 + 1) if v_max > v_min else 50
        for k in _etf_keys:
            if k not in result:
                result[k] = 50
        return result

    # Intentar importar Numba dentro del worker (necesario en spawn)
    try:
        from gpu_bt.cache import (calcular_perfs_nb, calcular_rs_scores_nb,
                              calcular_vol_rel_nb, calcular_vol_avg_nb,
                              calcular_h52_l52_nb, calcular_vol_sem_nb,
                              calcular_calidad_vela_nb, _PESOS_NB,
                              preparar_arrays_nb, get_idx_nb)
        _nb = True
        np_hist = preparar_arrays_nb(historico)
        spy_closes_nb = spy_df["Close"].to_numpy(dtype=_np.float64) if spy_df is not None else _np.array([], dtype=_np.float64)
        spy_sma_nb    = spy_sma.to_numpy(dtype=_np.float64) if spy_sma is not None else _np.array([], dtype=_np.float64)
    except Exception:
        _nb = False
        np_hist = {}
        spy_closes_nb = spy_sma_nb = None

    señales = []

    for fecha in fechas_chunk:
        ts_fecha  = pd.Timestamp(fecha)
        fecha_str = str(fecha)

        # ── Filtro de mercado ─────────────────────────────────────────────────
        mercado_alcista = True
        if p.filtro_mercado and spy_df is not None and spy_sma is not None:
            try:
                i_spy = spy_df.index.get_indexer([ts_fecha], method="pad")[0]
                if i_spy >= 0:
                    sp = float(spy_df.iloc[i_spy]["Close"])
                    sm = float(spy_sma.iloc[i_spy])
                    mercado_alcista = sp > sm
            except Exception:
                pass

        # ── RS de sectores para esta fecha (replica analyzer.get_sector_rs) ──
        sector_rs_fecha = _get_sector_rs_fecha(fecha_str)

        # ── Construir universo PIT con performances ───────────────────────────
        rows = []
        for t in tickers:
            if t in p.excluir_tickers:
                continue
            df = historico.get(t)
            if df is None or len(df) < _WARMUP_MIN + 5:
                continue
            try:
                if _nb and t in np_hist:
                    # ── Ruta rápida Numba ─────────────────────────────────────
                    arrs    = np_hist[t]
                    i       = get_idx_nb(np_hist, t, fecha_str)
                    closes  = arrs["closes"]
                    volumes = arrs["volumes"]
                    highs_a = arrs["highs"]
                    lows_a  = arrs["lows"]

                    if i < _WARMUP_MIN or i >= len(closes) - 1:
                        continue
                    c0 = closes[i]
                    if c0 <= 0:
                        continue

                    p1w, p1m, p3m, p6m, p1y = calcular_perfs_nb(closes, i)
                    vol_avg = calcular_vol_avg_nb(volumes, i)
                    rvol    = calcular_vol_rel_nb(volumes, i)
                    h52, l52 = calcular_h52_l52_nb(highs_a, lows_a, i)
                    pct52   = ((c0 / h52) - 1) * 100 if h52 > 0 else -99.0

                else:
                    # ── Ruta fallback pandas ──────────────────────────────────
                    i = df.index.get_indexer([ts_fecha], method="pad")[0]
                    if i < _WARMUP_MIN or i >= len(df) - 1:
                        continue
                    c0 = float(df.iloc[i]["Close"])
                    if c0 <= 0:
                        continue
                    p1w = _perf(df, i, 5) or 0.0
                    p1m = _perf(df, i, 21) or 0.0
                    p3m = _perf(df, i, 63) or 0.0
                    p6m = _perf(df, i, 126) or 0.0
                    p1y = _perf(df, i, 252) or 0.0
                    vol_avg = float(df.iloc[max(0, i-20):i]["Volume"].mean())
                    vol_hoy = float(df.iloc[i]["Volume"])
                    rvol    = vol_hoy / vol_avg if vol_avg > 0 else 1.0
                    i52 = max(0, i - 252)
                    h52 = float(df.iloc[i52:i+1]["High"].max())
                    l52 = float(df.iloc[i52:i+1]["Low"].min())
                    pct52 = ((c0 / h52) - 1) * 100 if h52 > 0 else -99.0

                rows.append({
                    "ticker":   t,
                    "price":    c0,
                    "vol_avg":  vol_avg,
                    "rel_vol":  rvol,
                    "h52":      h52,
                    "l52":      l52,
                    "pct52":    round(pct52, 1),
                    "_df_idx":  i,
                    "Perf1W": p1w, "Perf1M": p1m, "Perf3M": p3m,
                    "Perf6M": p6m, "Perf1Y": p1y,
                })
            except Exception:
                continue

        if len(rows) < 5:
            continue

        # ── RS Rating: réplica exacta de analyzer.calcular_rs_rating() ────────
        if _nb and len(rows) >= 5:
            perfs_mat = _np.array([
                [r["Perf1W"], r["Perf1M"], r["Perf3M"], r["Perf6M"], r["Perf1Y"]]
                for r in rows], dtype=_np.float64)
            rs_array = calcular_rs_scores_nb(perfs_mat, _PESOS_NB)
            for idx, row in enumerate(rows):
                row["RS_Rating"] = int(round(rs_array[idx]))
        else:
            periodos = list(_PESOS_RS.keys())
            for periodo in periodos:
                vals = [float(r.get(periodo) or 0) for r in rows]
                n = len(vals)
                orden = sorted(range(n), key=lambda x: vals[x])
                ranks = [0.0] * n
                for rk, orig_i in enumerate(orden):
                    ranks[orig_i] = (rk / (n - 1)) * 100 if n > 1 else 50.0
                for idx, row in enumerate(rows):
                    row[f"_rk_{periodo}"] = ranks[idx]
            for row in rows:
                row["_rs_score"] = sum(row[f"_rk_{p_}"] * w for p_, w in _PESOS_RS.items())
            rs_scores = [r["_rs_score"] for r in rows]
            s_min, s_max = min(rs_scores), max(rs_scores)
            for row in rows:
                if s_max > s_min:
                    row["RS_Rating"] = round(((row["_rs_score"] - s_min) / (s_max - s_min)) * 98 + 1)
                else:
                    row["RS_Rating"] = 50
            for row in rows:
                for p_ in list(_PESOS_RS.keys()):
                    row.pop(f"_rk_{p_}", None)
                row.pop("_rs_score", None)

        # ── Filtros + Estado + Señal ──────────────────────────────────────────
        for row in rows:
            rs     = row["RS_Rating"]
            avgvol = row["vol_avg"]
            pct52  = row["pct52"]
            rvol   = row["rel_vol"]
            t      = row["ticker"]

            # Filtros base (réplica de aplicar_filtros en analyzer.py)
            if rs < p.rs_minimo:
                continue
            if avgvol < p.vol_minimo_m * 1_000_000:
                continue
            if pct52 < p.pct_max_52w:
                continue
            if p.excluir_rs_90_95 and 90 <= rs < 95:
                continue

            p1w = float(row.get("Perf1W") or 0)
            p1m = float(row.get("Perf1M") or 0)
            p3m = float(row.get("Perf3M") or 0)

            # ── clasificar_estado — réplica exacta de analyzer.py ─────────────
            if p1w > 8 and rs >= 85 and rvol >= 1.5:
                estado = "BREAKOUT"
            elif abs(p1m) < 6 and p3m > 15 and rs >= 75:
                estado = "SQUEEZE"
            elif abs(p1m) < 8 and p3m > 20:
                estado = "EN BASE"
            elif p1m > 20:
                estado = "EXTENDIDO"
            elif p1m > 8:
                estado = "VIGILANCIA"
            else:
                continue   # SIN SEÑAL

            # ── Filtro volumen relativo por estado ────────────────────────────
            if estado == "BREAKOUT" and rvol < p.rvol_minimo:
                continue

            # ── Calidad de vela con OHLC exacto (replica calcular_calidad_vela) ─
            df      = historico[t]
            i       = row["_df_idx"]
            hi_hoy  = float(df.iloc[i]["High"])
            lo_hoy  = float(df.iloc[i]["Low"])
            op_hoy  = float(df.iloc[i]["Open"])
            c0      = row["price"]
            rango   = hi_hoy - lo_hoy
            cuerpo  = abs(c0 - op_hoy)

            ratio_cuerpo = cuerpo / rango if rango > 0 else 0
            pos_cierre   = (c0 - lo_hoy) / rango if rango > 0 else 0.5
            alcista      = c0 > op_hoy

            calidad_vela = round(
                ratio_cuerpo * 40 +
                pos_cierre   * 30 +
                (15 if alcista else 0) +
                (min(rvol, 3) / 3 * 15),
                1)

            if p.calidad_vela_min > 0 and calidad_vela < p.calidad_vela_min:
                continue

            # ── Proxy de earnings (gap >12% en próximos 5 días) ───────────────
            # NOTA: El broker usa otra estrategia (entra + stop a breakeven).
            # Son complementarias — el backtest no tiene fechas de earnings.
            gap_max = 0.0
            if p.filtro_earnings and i + 6 < len(df):
                for fwd in range(1, 6):
                    try:
                        o_f = float(df.iloc[i + fwd]["Open"])
                        c_a = float(df.iloc[i + fwd - 1]["Close"])
                        if c_a > 0:
                            gap_max = max(gap_max, abs(o_f / c_a - 1) * 100)
                    except Exception:
                        pass
                if gap_max > 12.0:
                    continue

            # ── Score de confianza ────────────────────────────────────────────
            h52 = row["h52"]
            l52 = row["l52"]
            acelerando    = (p1w * 4.3) > p1m and p1m > (p3m / 3 * 0.7)
            base_estrecha = abs(p1m) < 8 and p3m > 15

            vol_score  = 100 if rvol >= 2.5 else 70 if rvol >= 1.5 else 40
            mom_score  = min(100, max(0,
                min(p1w * 4.3, 20) * 1.5 + min(p1m, 15) * 1.5 +
                (10 if acelerando else 0)))
            pos_rango  = (c0 - l52) / max(h52 - l52, 0.01)
            estr_score = min(100,
                pos_rango * 40 +
                (15 if pct52 >= -5 else 8 if pct52 >= -10 else 0) +
                (20 if base_estrecha else 0))

            score = round(min(100, max(0,
                rs * 0.35 + vol_score * 0.20 + mom_score * 0.20 +
                estr_score * 0.15 + calidad_vela * 0.10
            )), 1)

            if score < p.score_minimo:
                continue

            # Régimen de volatilidad
            ret_sem = []
            for w in range(min(12, i // 5)):
                p0s = float(df.iloc[i - (w + 1) * 5]["Close"])
                p1s = float(df.iloc[i - w * 5]["Close"])
                if p0s > 0:
                    ret_sem.append(abs(p1s / p0s - 1) * 100)
            vol_sem = _stats.mean(ret_sem) if ret_sem else 3.0
            regimen = ("tranquilo" if vol_sem < 2 else
                       "normal"    if vol_sem < 4 else
                       "alto"      if vol_sem < 7 else "extremo")

            if p.penalizar_vol_alta and regimen == "alto" and score < p.score_minimo + p.score_vol_extra:
                continue

            # ── calcular_posicion — réplica exacta de analyzer.py ─────────────
            if estado == "BREAKOUT":
                entrada_t = round(c0 * 0.97, 2)
            elif estado in ("EN BASE", "SQUEEZE"):
                entrada_t = round(c0 * 1.001, 2)
            elif estado == "EXTENDIDO":
                entrada_t = round(c0 * (1 - p.stop_pct * 0.8), 2)
            else:  # VIGILANCIA
                entrada_t = round(c0 * 0.98, 2)

            # Stop con volatilidad_factor (igual que analyzer.py)
            volatilidad_factor = 1.0 + abs(p1w) / 200
            stop_dist  = entrada_t * p.stop_pct * volatilidad_factor
            stop       = round(entrada_t - stop_dist, 2)
            riesgo_pp  = entrada_t - stop
            t1_p       = round(entrada_t + riesgo_pp * p.target_rr_1, 2)
            t2_p       = round(entrada_t + riesgo_pp * p.target_rr_2, 2)

            # Entrada real = Open del día siguiente
            try:
                entrada_real     = float(df.iloc[i + 1]["Open"])
                fecha_entrada    = df.index[i + 1].date()
            except Exception:
                continue
            if entrada_real <= 0:
                continue

            señales.append({
                "fecha":           str(fecha),
                "fecha_entrada":   str(fecha_entrada),
                "ticker":          t,
                "estado":          estado,
                "rs":              rs,
                "score":           score,
                "regimen":         regimen,
                "rel_vol":         round(rvol, 2),
                "calidad_vela":    calidad_vela,
                "gap_earnings":    round(gap_max, 1),
                "pct_52w":         round(pct52, 1),
                "perf_1w":         round(p1w, 1),
                "perf_1m":         round(p1m, 1),
                "perf_3m":         round(p3m, 1),
                "precio_señal":    round(c0, 2),
                "entrada_teorica": entrada_t,
                "entrada_real":    round(entrada_real, 2),
                "stop":            stop,
                "t1":              t1_p,
                "t2":              t2_p,
                "vol_semanal":     round(vol_sem, 2),
                "mercado_alcista": mercado_alcista,
                "h52":             round(h52, 2),
                "l52":             round(l52, 2),
            })

    return señales


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE SEÑALES (PARALELA)
# ═══════════════════════════════════════════════════════════════════════════════

def construir_señales(historico: dict, desde: datetime.date,
                      hasta: datetime.date, params: Params,
                      n_workers: int = None) -> list:
    """Construye señales para el período dado usando paralelismo por fechas."""
    spy_df = historico.get("SPY")
    cal_df = spy_df if spy_df is not None else next(iter(historico.values()))
    fechas = [d.date() for d in cal_df.index if desde <= d.date() <= hasta]

    if not fechas:
        return []

    n_cores = n_workers or multiprocessing.cpu_count()
    hkeys, hdata = serializar_historico(historico)
    params_dict  = params.as_dict()
    params_dict["excluir_tickers"] = list(params.excluir_tickers)

    fechas_work = fechas[:-1]
    chunk_size  = max(1, len(fechas_work) // n_cores)
    chunks      = [fechas_work[i:i + chunk_size]
                   for i in range(0, len(fechas_work), chunk_size)]

    worker_args = [(chunk, hkeys, hdata, params_dict) for chunk in chunks]

    if n_cores == 1:
        resultados = [_escanear_chunk(a) for a in worker_args]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_cores) as pool:
            resultados = pool.map(_escanear_chunk, worker_args)

    señales = [s for chunk in resultados for s in chunk]
    # Desempate por ticker: sort solo-por-fecha es estable pero deja el orden
    # de empates a merced del orden de iteración del histórico, que es
    # no-determinista (list(set()) sin PYTHONHASHSEED). Con cap de slots en
    # simular_portfolio, ese orden cambia qué señales se ejecutan → resultados
    # WF distintos entre runs. (fecha, ticker) lo hace 100% reproducible.
    señales.sort(key=lambda x: (x["fecha"], x["ticker"]))
    return señales


# ═══════════════════════════════════════════════════════════════════════════════
#  PRE-CÓMPUTO DE CANDIDATOS
#  Separa el escaneo del histórico (caro) del filtrado por params (barato).
#  El walk-forward llama a precomputar_candidatos UNA vez por ventana,
#  luego filtrar_candidatos N veces (una por combinación de params).
# ═══════════════════════════════════════════════════════════════════════════════

def _precomputar_chunk(args: tuple) -> list:
    """Worker para precomputar_candidatos. Extrae TODOS los atributos sin filtrar."""
    (fechas_chunk, historico_keys, historico_data) = args

    import statistics as _stats
    historico  = _deserializar_historico(historico_keys, historico_data)
    spy_df     = historico.get("SPY")
    spy_sma200 = _spy_sma(spy_df, 200)
    tickers    = [t for t in historico if t != "SPY" and not t.startswith("ETF_")]

    candidatos = []

    for fecha in fechas_chunk:
        ts_fecha = pd.Timestamp(fecha)

        # Mercado
        mercado_alcista = True
        if spy_df is not None and spy_sma200 is not None:
            try:
                i_spy = spy_df.index.get_indexer([ts_fecha], method="pad")[0]
                if i_spy >= 0:
                    mercado_alcista = (float(spy_df.iloc[i_spy]["Close"]) >
                                       float(spy_sma200.iloc[i_spy]))
            except Exception:
                pass

        # Construir universo PIT
        rows = []
        for t in tickers:
            df = historico.get(t)
            if df is None or len(df) < _WARMUP_MIN + 5:
                continue
            try:
                i = df.index.get_indexer([ts_fecha], method="pad")[0]
                if i < _WARMUP_MIN or i >= len(df) - 1:
                    continue
                c0 = float(df.iloc[i]["Close"])
                if c0 <= 0:
                    continue
                perfs = {}
                for nombre, ventana in _VENTANAS_RS.items():
                    v = _perf(df, i, ventana)
                    perfs[nombre] = v if v is not None else 0.0
                vol_avg = float(df.iloc[max(0, i-20):i]["Volume"].mean())
                vol_hoy = float(df.iloc[i]["Volume"])
                rel_vol = vol_hoy / vol_avg if vol_avg > 0 else 1.0
                vol_60d = float(df.iloc[max(0, i-60):i]["Volume"].mean())
                i52  = max(0, i - 252)
                h52  = float(df.iloc[i52:i+1]["High"].max())
                l52  = float(df.iloc[i52:i+1]["Low"].min())
                pct52 = ((c0 / h52) - 1) * 100 if h52 > 0 else -99.0
                rows.append({
                    "ticker":   t,
                    "price":    c0,
                    "vol_avg":  vol_avg,
                    "vol_60d":  vol_60d,
                    "rel_vol":  rel_vol,
                    "h52":      h52,
                    "l52":      l52,
                    "pct52":    round(pct52, 1),
                    "_df_idx":  i,
                    **perfs,
                })
            except Exception:
                continue

        if len(rows) < 5:
            continue

        # RS Rating exacto (5 períodos, pesos de analyzer.py)
        periodos = list(_PESOS_RS.keys())
        for periodo in periodos:
            vals = [float(r.get(periodo) or 0) for r in rows]
            n = len(vals)
            orden = sorted(range(n), key=lambda x: vals[x])
            ranks = [0.0] * n
            for rk, orig_i in enumerate(orden):
                ranks[orig_i] = (rk / (n - 1)) * 100 if n > 1 else 50.0
            for idx, row in enumerate(rows):
                row[f"_rk_{periodo}"] = ranks[idx]
        for row in rows:
            row["_rs_score"] = sum(row[f"_rk_{p_}"] * w for p_, w in _PESOS_RS.items())
        rs_scores = [r["_rs_score"] for r in rows]
        s_min, s_max = min(rs_scores), max(rs_scores)
        for row in rows:
            if s_max > s_min:
                row["RS_Rating"] = round(
                    ((row["_rs_score"] - s_min) / (s_max - s_min)) * 98 + 1)
            else:
                row["RS_Rating"] = 50
        for row in rows:
            for p_ in periodos:
                row.pop(f"_rk_{p_}", None)
            row.pop("_rs_score", None)

        # Extraer candidatos con todos sus atributos
        for row in rows:
            rs     = row["RS_Rating"]
            rvol   = row["rel_vol"]
            c0     = row["price"]
            pct52  = row["pct52"]
            p1w    = float(row.get("Perf1W") or 0)
            p1m    = float(row.get("Perf1M") or 0)
            p3m    = float(row.get("Perf3M") or 0)
            h52    = row["h52"]
            l52    = row["l52"]
            t      = row["ticker"]
            i      = row["_df_idx"]
            df     = historico[t]

            # Estado pre-clasificado (con umbrales de analyzer.py)
            if p1w > 8 and rs >= 85 and rvol >= 1.5:
                estado = "BREAKOUT"
            elif abs(p1m) < 6 and p3m > 15 and rs >= 75:
                estado = "SQUEEZE"
            elif abs(p1m) < 8 and p3m > 20:
                estado = "EN BASE"
            elif p1m > 20:
                estado = "EXTENDIDO"
            elif p1m > 8:
                estado = "VIGILANCIA"
            else:
                continue

            # OHLC para calidad de vela
            try:
                hi_h = float(df.iloc[i]["High"])
                lo_h = float(df.iloc[i]["Low"])
                op_h = float(df.iloc[i]["Open"])
                rango  = hi_h - lo_h
                cuerpo = abs(c0 - op_h)
                ratio_cuerpo = cuerpo / rango if rango > 0 else 0
                pos_cierre   = (c0 - lo_h) / rango if rango > 0 else 0.5
                calidad_vela = round(
                    ratio_cuerpo * 40 + pos_cierre * 30 +
                    (15 if c0 > op_h else 0) +
                    (min(rvol, 3) / 3 * 15), 1)
            except Exception:
                calidad_vela = 0.0

            # Earnings proxy
            gap_max = 0.0
            if i + 6 < len(df):
                for fwd in range(1, 6):
                    try:
                        o_f = float(df.iloc[i + fwd]["Open"])
                        c_a = float(df.iloc[i + fwd - 1]["Close"])
                        if c_a > 0:
                            gap_max = max(gap_max, abs(o_f / c_a - 1) * 100)
                    except Exception:
                        pass

            # Régimen de volatilidad
            ret_sem = []
            for w in range(min(12, i // 5)):
                p0s = float(df.iloc[i - (w + 1) * 5]["Close"])
                p1s = float(df.iloc[i - w * 5]["Close"])
                if p0s > 0:
                    ret_sem.append(abs(p1s / p0s - 1) * 100)
            vol_sem = _stats.mean(ret_sem) if ret_sem else 3.0
            regimen = ("tranquilo" if vol_sem < 2 else
                       "normal"    if vol_sem < 4 else
                       "alto"      if vol_sem < 7 else "extremo")

            # Sub-scores para filtrar_candidatos
            acelerando    = (p1w * 4.3) > p1m and p1m > (p3m / 3 * 0.7)
            base_estrecha = abs(p1m) < 8 and p3m > 15
            vol_score     = 100 if rvol >= 2.5 else 70 if rvol >= 1.5 else 40
            mom_score     = min(100, max(0,
                min(p1w * 4.3, 20) * 1.5 + min(p1m, 15) * 1.5 +
                (10 if acelerando else 0)))
            pos_rango     = (c0 - l52) / max(h52 - l52, 0.01)
            estr_score    = min(100,
                pos_rango * 40 +
                (15 if pct52 >= -5 else 8 if pct52 >= -10 else 0) +
                (20 if base_estrecha else 0))

            # Entrada real
            try:
                entrada_real  = float(df.iloc[i + 1]["Open"])
                fecha_entrada = str(df.index[i + 1].date())
            except Exception:
                continue
            if entrada_real <= 0:
                continue

            candidatos.append({
                "fecha":         str(fecha),
                "fecha_entrada": fecha_entrada,
                "ticker":        t,
                "rs":            rs,
                "p1w":           round(p1w, 2),
                "p1m":           round(p1m, 2),
                "p3m":           round(p3m, 2),
                "pct52":         round(pct52, 2),
                "rel_vol":       round(rvol, 2),
                "vol_avg_m":     round(row["vol_avg"] / 1e6, 3),
                "vol_60d":       round(row["vol_60d"], 0),
                "gap_earnings":  round(gap_max, 2),
                "calidad_vela":  calidad_vela,
                "vol_sem":       round(vol_sem, 2),
                "regimen":       regimen,
                "acelerando":    acelerando,
                "base_estrecha": base_estrecha,
                "estado":        estado,
                "mercado_alcista": mercado_alcista,
                "vol_score":     vol_score,
                "mom_score":     round(mom_score, 2),
                "estr_score":    round(estr_score, 2),
                "precio_señal":  round(c0, 4),
                "entrada_real":  round(entrada_real, 4),
                "h52":           round(h52, 4),
                "l52":           round(l52, 4),
            })

    return candidatos


def precomputar_candidatos(historico: dict,
                           desde: datetime.date,
                           hasta: datetime.date,
                           n_workers: int = None) -> list:
    """
    Escanea el histórico UNA sola vez en paralelo (por chunks de fechas)
    y devuelve todos los candidatos con sus atributos pre-calculados.
    """
    spy_df      = historico.get("SPY")
    cal_df      = spy_df if spy_df is not None else next(iter(historico.values()))
    fechas      = [d.date() for d in cal_df.index if desde <= d.date() <= hasta]
    fechas_work = fechas[:-1]
    if not fechas_work:
        return []

    n_cpu        = n_workers or multiprocessing.cpu_count()
    hkeys, hdata = serializar_historico(historico)
    chunk_size   = max(1, len(fechas_work) // n_cpu)
    chunks       = [fechas_work[i:i + chunk_size]
                    for i in range(0, len(fechas_work), chunk_size)]
    args         = [(chunk, hkeys, hdata) for chunk in chunks]

    if n_cpu == 1:
        resultados = [_precomputar_chunk(a) for a in args]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_cpu) as pool:
            resultados = pool.map(_precomputar_chunk, args)

    candidatos = [c for chunk in resultados for c in chunk]
    # Desempate por ticker → orden 100% reproducible (ver nota en construir_señales)
    candidatos.sort(key=lambda x: (x["fecha"], x["ticker"]))
    return candidatos


def filtrar_candidatos(candidatos: list, params: Params,
                       estado_filtro: str = None) -> list:
    """
    Aplica los filtros de Params sobre candidatos pre-computados.
    Opera solo sobre dicts — sin DataFrames. Muy rápido.
    """
    señales = []

    for c in candidatos:
        rs      = c["rs"]
        rvol    = c["rel_vol"]
        estado  = c["estado"]
        regimen = c["regimen"]
        pct52   = c["pct52"]

        # Filtros base (réplica de aplicar_filtros en analyzer.py)
        if rs < params.rs_minimo:
            continue
        if c["vol_avg_m"] < params.vol_minimo_m:
            continue
        if pct52 < params.pct_max_52w:
            continue
        if params.excluir_rs_90_95 and 90 <= rs < 95:
            continue
        if params.filtro_mercado and not c["mercado_alcista"]:
            continue
        if params.filtro_earnings and c["gap_earnings"] > 12.0:
            continue
        if params.calidad_vela_min > 0 and c["calidad_vela"] < params.calidad_vela_min:
            continue

        # Estado re-evaluado con umbrales exactos de analyzer.py
        p1w = c["p1w"]
        p1m = c["p1m"]
        p3m = c["p3m"]
        if p1w > 8 and rs >= 85 and rvol >= 1.5:
            estado_real = "BREAKOUT"
        elif abs(p1m) < 6 and p3m > 15 and rs >= 75:
            estado_real = "SQUEEZE"
        elif abs(p1m) < 8 and p3m > 20:
            estado_real = "EN BASE"
        elif p1m > 20:
            estado_real = "EXTENDIDO"
        elif p1m > 8:
            estado_real = "VIGILANCIA"
        else:
            continue

        # Filtro por estado (CLI --estado)
        if estado_filtro and estado_filtro.upper() not in estado_real.upper():
            continue

        # Filtro rvol para BREAKOUT
        if estado_real == "BREAKOUT" and rvol < params.rvol_minimo:
            continue

        # Score de confianza
        score = round(min(100, max(0,
            rs * 0.35 + c["vol_score"] * 0.20 + c["mom_score"] * 0.20 +
            c["estr_score"] * 0.15 + c["calidad_vela"] * 0.10
        )), 1)

        if score < params.score_minimo:
            continue
        if params.penalizar_vol_alta and regimen == "alto" and score < params.score_minimo + params.score_vol_extra:
            continue

        # Niveles de operación (réplica exacta de calcular_posicion en analyzer.py)
        c0 = c["precio_señal"]
        if estado_real == "BREAKOUT":
            entrada_t = round(c0 * 0.97, 2)
        elif estado_real in ("EN BASE", "SQUEEZE"):
            entrada_t = round(c0 * 1.001, 2)
        elif estado_real == "EXTENDIDO":
            entrada_t = round(c0 * (1 - params.stop_pct * 0.8), 2)
        else:
            entrada_t = round(c0 * 0.98, 2)

        volatilidad_factor = 1.0 + abs(p1w) / 200
        stop_dist  = entrada_t * params.stop_pct * volatilidad_factor
        stop       = round(entrada_t - stop_dist, 2)
        riesgo_pp  = entrada_t - stop
        t1_p       = round(entrada_t + riesgo_pp * params.target_rr_1, 2)
        t2_p       = round(entrada_t + riesgo_pp * params.target_rr_2, 2)

        señales.append({
            **c,
            "estado":          estado_real,
            "score":           score,
            "entrada_teorica": entrada_t,
            "stop":            stop,
            "t1":              t1_p,
            "t2":              t2_p,
            "perf_1w":         p1w,
            "perf_1m":         p1m,
            "perf_3m":         p3m,
            "pct_52w":         pct52,
        })

    return señales


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN DE OPERACIÓN (parallelizable)
# ═══════════════════════════════════════════════════════════════════════════════

# Worker-local globals (inicializadas por _init_sim_worker en cada proceso hijo)
_sim_hist   = None
_sim_params = None


def _init_sim_worker(hkeys, hdata, params_dict):
    """Initializer del Pool: deserializa el histórico UNA vez por worker."""
    global _sim_hist, _sim_params
    _sim_hist   = _deserializar_historico(hkeys, hdata)
    p = params_dict.copy()
    p["excluir_tickers"] = tuple(p.get("excluir_tickers", []))
    _sim_params = Params(**p)


def _simular_operacion_worker(señal: dict) -> dict:
    """Wrapper sin args extra — el Pool solo puede mapear un argumento.
    A2/A3: dispatch via strategy registry. Worker-local cache de sim_fn."""
    global _sim_fn_cached
    if "_sim_fn_cached" not in globals() or _sim_fn_cached is None:
        try:
            from gpu_bt.strategies import get_strategy
            sid = getattr(_sim_params, "strategy_id", "baseline") or "baseline"
            _, _sim_fn_cached = get_strategy(sid)
        except Exception:
            _sim_fn_cached = simular_operacion
    return _sim_fn_cached(señal, _sim_hist, _sim_params)


_sim_fn_cached = None


def simular_operacion(señal: dict, historico: dict, params: Params) -> dict:
    """Simula el ciclo de vida completo de una operación."""
    ticker      = señal["ticker"]
    entrada_raw = señal["entrada_real"]
    entrada     = round(entrada_raw * (1 + params.slippage_pct), 2)
    stop        = señal["stop"]
    t1_p        = señal["t1"]
    t2_p        = señal["t2"]
    fecha_ent   = datetime.date.fromisoformat(señal["fecha_entrada"])
    df          = historico.get(ticker)
    spy         = historico.get("SPY")

    NO_DATA = {**señal, "outcome": "SIN_DATOS", "pnl_pct": 0.0,
               "dias_en_operacion": 0, "precio_salida": 0.0,
               "motivo_salida": "sin_datos", "t1_alcanzado": False,
               "entrada_efectiva": entrada}

    if df is None or entrada <= 0:
        return NO_DATA
    try:
        idx0 = df.index.get_indexer([pd.Timestamp(fecha_ent)], method="pad")[0]
        if idx0 < 0:
            return NO_DATA
    except Exception:
        return NO_DATA

    spy_sma       = _spy_sma(spy, params.spy_sma_periodo)
    stop_actual   = stop
    t1_hit        = False
    fraccion_viva = 1.0
    pnl_parcial   = 0.0
    precio_salida = entrada
    motivo        = "expirado"
    dias          = 0
    max_precio    = entrada

    # B3: trailing mode via strategy_params. Default "pct" = trailing % bajo
    # max post-T1 (comportamiento original). "atr_chandelier" = max -
    # atr_mult × ATR(atr_period). ATR pre-computado on-demand una vez.
    sp = getattr(params, "strategy_params", {}) or {}
    trailing_mode = sp.get("trailing_mode", "pct")
    atr_arr = None
    if trailing_mode == "atr_chandelier" and t1_p > 0:
        try:
            from gpu_bt.indicators import atr as _atr_calc
            atr_period = int(sp.get("atr_period", 22))
            atr_arr = _atr_calc(df["High"].values, df["Low"].values,
                                df["Close"].values, atr_period)
        except Exception:
            atr_arr = None

    for j in range(1, params.max_hold_days + 1):
        idx = idx0 + j
        if idx >= len(df):
            precio_salida = float(df.iloc[-1]["Close"])
            motivo = "fin_datos"; dias = j; break

        op_  = float(df.iloc[idx]["Open"])
        hi_  = float(df.iloc[idx]["High"])
        lo_  = float(df.iloc[idx]["Low"])
        cl_  = float(df.iloc[idx]["Close"])
        fec_ = df.index[idx].date()
        dias = j

        # Filtro mercado intraoperación eliminado — el bot (PositionMonitor) no
        # cierra posiciones por SPY < SMA200 durante la operación. El backtest
        # debe replicar exactamente el comportamiento del bot.

        # Stop (gap adverso → fill al peor precio posible, conservador vs broker real)
        if lo_ <= stop_actual:
            precio_salida = min(op_, stop_actual); motivo = "stop"; break

        # T1: cierre parcial + stop a breakeven (+ trailing si activo)
        if not t1_hit and hi_ >= t1_p:
            t1_hit        = True
            fraccion_viva = 1.0 - params.fraccion_t1
            pnl_parcial   = (t1_p - entrada) / entrada * params.fraccion_t1
            stop_actual   = max(stop_actual, entrada)
            max_precio    = hi_
            if hi_ >= t2_p:
                precio_salida = t2_p; motivo = "t2"; break
            continue

        # Trailing stop post-T1
        if t1_hit:
            if hi_ > max_precio:
                max_precio = hi_
            trail = None
            if trailing_mode == "atr_chandelier" and atr_arr is not None:
                # Chandelier: stop = max_precio - mult × ATR(idx)
                atr_mult = float(sp.get("atr_mult", 3.0))
                a = atr_arr[idx] if idx < len(atr_arr) else None
                if a is not None and not np.isnan(a) and a > 0:
                    trail = max_precio - atr_mult * a
            elif params.trailing_stop_pct > 0:
                trail = max_precio * (1.0 - params.trailing_stop_pct)
            if trail is not None and trail > stop_actual:
                stop_actual = trail

        # T2
        if t1_hit and hi_ >= t2_p:
            precio_salida = t2_p; motivo = "t2"; break

        if j == params.max_hold_days:
            precio_salida = cl_; motivo = "expirado"

    pnl_total = pnl_parcial + (precio_salida - entrada) / entrada * fraccion_viva
    pnl_pct   = round(pnl_total * 100, 2)

    if   motivo == "stop":   outcome = "LOSS"
    elif motivo == "t2":     outcome = "WIN_T2"
    elif t1_hit:             outcome = "WIN_T1"
    elif pnl_pct >  0.5:    outcome = "WIN_PARCIAL"
    elif pnl_pct < -0.5:    outcome = "LOSS"
    else:                    outcome = "BREAKEVEN"

    return {**señal,
            "outcome":           outcome,
            "pnl_pct":           pnl_pct,
            "dias_en_operacion": dias,
            "precio_salida":     round(precio_salida, 2),
            "motivo_salida":     motivo,
            "t1_alcanzado":      t1_hit,
            "stop_final":        round(stop_actual, 2),
            "entrada_efectiva":  entrada,
            "slippage_usd":      round(entrada - entrada_raw, 3)}


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN DE PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

def simular_portfolio(operaciones: list, params: Params) -> dict:
    """
    Simula gestión real del portfolio:
      - MAX_POSICIONES simultáneas como máximo
      - Sizing: RIESGO_POR_OP% del capital actual por operación
    """
    ops_validas = [o for o in operaciones if o["outcome"] != "SIN_DATOS"]
    # Desempate por ticker: con cap max_posiciones, las ops de la misma fecha
    # compiten por slot. Sort solo-por-fecha deja el desempate al orden de
    # iteración no-determinista → resultados distintos entre runs. (fecha, ticker)
    # lo hace reproducible.
    ops_validas.sort(key=lambda x: (x["fecha_entrada"], x["ticker"]))

    capital       = params.capital_total
    posiciones    = []
    ejecutadas    = []
    n_descartadas = 0

    for op in ops_validas:
        fecha_ent = op["fecha_entrada"]
        posiciones = [p for p in posiciones if p["fecha_salida"] > fecha_ent]

        if len(posiciones) >= params.max_posiciones:
            n_descartadas += 1; continue

        entrada   = op.get("entrada_efectiva", op["entrada_real"])
        stop      = op["stop"]
        riesgo_pp = max(entrada - stop, 0.01)
        riesgo_usd = capital * params.riesgo_por_op
        # B10 heat cap: si la suma de riesgos abiertos + nuevo excede
        # heat_cap × capital, skip. heat_cap=1.0 (default) = sin cap.
        heat_cap = float(getattr(params, "heat_cap", 1.0) or 1.0)
        if heat_cap < 1.0:
            heat_actual = sum(p.get("riesgo_usd", 0.0) for p in posiciones)
            if (heat_actual + riesgo_usd) > capital * heat_cap:
                n_descartadas += 1
                continue
        n_acc     = max(1, int(riesgo_usd / riesgo_pp))
        coste     = n_acc * entrada
        if coste > capital * 0.40:
            n_acc = max(1, int(capital * 0.40 / entrada))
            coste = n_acc * entrada

        if op.get("t1_alcanzado"):
            t1_price  = op.get("t1", entrada)
            sal_price = op.get("precio_salida", entrada)
            pnl_usd = (n_acc * params.fraccion_t1 * (t1_price - entrada) +
                       n_acc * (1 - params.fraccion_t1) * (sal_price - entrada))
        else:
            pnl_usd = n_acc * (op.get("precio_salida", entrada) - entrada)

        capital += pnl_usd

        dias = op.get("dias_en_operacion", params.max_hold_days)
        try:
            fd = datetime.date.fromisoformat(fecha_ent)
            fs = str(fd + datetime.timedelta(days=dias + 2))
        except Exception:
            fs = fecha_ent

        posiciones.append({"ticker": op["ticker"], "fecha_salida": fs,
                           "riesgo_usd": riesgo_usd})
        ejecutadas.append({**op,
            "n_acciones":   n_acc,
            "pnl_usd":      round(pnl_usd, 2),
            "capital_post": round(capital, 2)})

    if not ejecutadas:
        return {"error": "Sin operaciones ejecutadas"}

    pnls       = [o["pnl_usd"] for o in ejecutadas]
    wins_usd   = [p for p in pnls if p > 0]
    losses_usd = [p for p in pnls if p < 0]
    n_wins     = len([o for o in ejecutadas if "WIN" in o["outcome"]])
    n_losses   = len([o for o in ejecutadas if o["outcome"] == "LOSS"])
    n_valid    = n_wins + n_losses

    # Equity curve + max_dd con duración (días bajo agua máximo)
    eq = params.capital_total; peak = eq; max_dd = 0.0
    peak_date = None
    max_dd_duration_days = 0
    cur_dd_start_date = None
    equity_curve = []  # list[(fecha_iso, equity)]
    for o in ejecutadas:
        eq += o["pnl_usd"]
        fd_iso = o.get("fecha_entrada", "")
        try:
            fd_dt = datetime.date.fromisoformat(fd_iso)
        except Exception:
            fd_dt = None
        if eq >= peak:
            peak = eq
            peak_date = fd_dt
            cur_dd_start_date = None
        else:
            if cur_dd_start_date is None:
                cur_dd_start_date = peak_date or fd_dt
            if fd_dt and cur_dd_start_date:
                dur = (fd_dt - cur_dd_start_date).days
                if dur > max_dd_duration_days:
                    max_dd_duration_days = dur
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd_pct)
        equity_curve.append((fd_iso, round(eq, 2)))

    retorno = (capital - params.capital_total) / params.capital_total * 100
    pf = (sum(wins_usd) / abs(sum(losses_usd))
          if losses_usd and sum(losses_usd) != 0 else 0.0)

    # ─── A4: Métricas avanzadas ──────────────────────────────────────────────
    # Sharpe / Sortino / Calmar sobre retornos diarios derivados de equity curve.
    # Asume 252 days/year. Risk-free = 0 (paper trading).
    sharpe = sortino = calmar = 0.0
    if len(equity_curve) >= 2:
        eq_values = [e for _, e in equity_curve]
        # Returns por operación (proxy daily). Para Sharpe annualized:
        # asumimos avg_hold_days = total_days / n_ops aprox 1 — usamos
        # op-level returns que son ya múltiples días. Annualization
        # factor = 252 / avg_hold_days. Sin avg hold real, factor=12
        # (mensual aprox) es conservador.
        try:
            avg_hold = max(1, sum(o.get("dias_en_operacion", 1) for o in ejecutadas) / len(ejecutadas))
        except Exception:
            avg_hold = 1
        ann_factor = (252.0 / avg_hold) ** 0.5
        # Op-level returns (delta equity)
        op_rets = []
        for i in range(1, len(eq_values)):
            prev = eq_values[i - 1]
            if prev > 0:
                op_rets.append((eq_values[i] - prev) / prev)
        if op_rets:
            mean_r = statistics.mean(op_rets)
            std_r = statistics.pstdev(op_rets) if len(op_rets) > 1 else 0.0
            neg_r = [r for r in op_rets if r < 0]
            downside_std = statistics.pstdev(neg_r) if len(neg_r) > 1 else 0.0
            if std_r > 0:
                sharpe = round((mean_r / std_r) * ann_factor, 3)
            if downside_std > 0:
                sortino = round((mean_r / downside_std) * ann_factor, 3)
    # Calmar = retorno_anual / max_dd. Estima retorno anual desde total
    if max_dd > 0:
        # Asumir período backtest cubre n años aproximado por equity_curve
        try:
            first_d = datetime.date.fromisoformat(equity_curve[0][0])
            last_d = datetime.date.fromisoformat(equity_curve[-1][0])
            n_years = max(0.1, (last_d - first_d).days / 365.25)
            ret_ann = ((capital / params.capital_total) ** (1 / n_years) - 1) * 100
        except Exception:
            ret_ann = retorno
        calmar = round(ret_ann / max_dd, 3) if max_dd > 0 else 0.0

    # Expectancy = avg_win × win_rate - avg_loss × loss_rate (PnL esperado por op)
    avg_win_pct = (statistics.mean([o["pnl_usd"] / params.capital_total * 100 for o in ejecutadas
                                     if "WIN" in o["outcome"]]) if n_wins else 0)
    avg_loss_pct = (statistics.mean([o["pnl_usd"] / params.capital_total * 100 for o in ejecutadas
                                      if o["outcome"] == "LOSS"]) if n_losses else 0)
    wr_frac = n_wins / n_valid if n_valid else 0
    expectancy_pct = round(avg_win_pct * wr_frac + avg_loss_pct * (1 - wr_frac), 4)

    # Streaks consecutivos
    cur_win_streak = max_win_streak = 0
    cur_loss_streak = max_loss_streak = 0
    for o in ejecutadas:
        if "WIN" in o["outcome"]:
            cur_win_streak += 1
            cur_loss_streak = 0
            max_win_streak = max(max_win_streak, cur_win_streak)
        elif o["outcome"] == "LOSS":
            cur_loss_streak += 1
            cur_win_streak = 0
            max_loss_streak = max(max_loss_streak, cur_loss_streak)

    return {
        "capital_inicial":    params.capital_total,
        "capital_final":      round(capital, 2),
        "retorno_pct":        round(retorno, 2),
        "max_drawdown_pct":   round(max_dd, 2),
        "max_dd_duration_d":  max_dd_duration_days,
        "win_rate_pct":       round(n_wins / n_valid * 100, 1) if n_valid else 0,
        "profit_factor":      round(pf, 3),
        "sharpe":             sharpe,
        "sortino":            sortino,
        "calmar":             calmar,
        "expectancy_pct":     expectancy_pct,
        "avg_win_usd":        round(statistics.mean(wins_usd), 2) if wins_usd else 0,
        "avg_loss_usd":       round(statistics.mean(losses_usd), 2) if losses_usd else 0,
        "max_win_streak":     max_win_streak,
        "max_loss_streak":    max_loss_streak,
        "n_ejecutadas":       len(ejecutadas),
        "n_descartadas":      n_descartadas,
        "n_wins":             n_wins,
        "n_losses":           n_losses,
        "expectativa_usd":    round(statistics.mean(pnls), 2) if pnls else 0,
        "equity_curve":       equity_curve,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_rapido(candidatos: list, historico: dict,
                        params: Params, estado_filtro: str = None) -> dict:
    """
    Versión rápida: usa candidatos pre-computados.
    Evita re-escanear el histórico — solo filtra y simula.

    A2/A3: dispatch via strategy registry (params.strategy_id).
    Default "baseline" = comportamiento original (analyzer.py).
    """
    # Strategy plugin dispatch
    try:
        from gpu_bt.strategies import get_strategy
        strategy_id = getattr(params, "strategy_id", "baseline") or "baseline"
        filter_fn, sim_fn = get_strategy(strategy_id)
    except Exception:
        # Fallback inline si registry no carga
        filter_fn = lambda c, p, ef=None: filtrar_candidatos(c, p, ef)
        sim_fn = lambda s, h, p: simular_operacion(s, h, p)

    try:
        señales = filter_fn(candidatos, params, estado_filtro, historico=historico)
    except TypeError:
        try:
            señales = filter_fn(candidatos, params, estado_filtro)
        except TypeError:
            señales = filter_fn(candidatos, params)
    if not señales:
        return {"error": "Sin señales", "n_señales": 0,
                "params": params.as_dict(), "n_operaciones": 0}

    operaciones = [sim_fn(s, historico, params) for s in señales]
    pf = simular_portfolio(operaciones, params)

    validas = [o for o in operaciones if o["outcome"] != "SIN_DATOS"]
    wins    = [o for o in validas if "WIN" in o["outcome"]]
    losses  = [o for o in validas if o["outcome"] == "LOSS"]
    nv      = len(wins) + len(losses)
    pnls    = [o["pnl_pct"] for o in validas]
    pw      = [o["pnl_pct"] for o in wins]
    pl      = [o["pnl_pct"] for o in losses]

    wr_ops  = len(wins) / nv * 100 if nv else 0
    pf_ops  = sum(pw) / abs(sum(pl)) if pl and sum(pl) != 0 else 0
    avg_pnl = statistics.mean(pnls) if pnls else 0

    eq = peak = 100.0; max_dd_ops = 0.0
    for o in sorted(validas, key=lambda x: x["fecha"]):
        eq   *= 1 + o["pnl_pct"] / 100
        peak  = max(peak, eq)
        max_dd_ops = max(max_dd_ops, (peak - eq) / peak * 100)

    return {
        "params":           params.as_dict(),
        "params_label":     params.label(),
        "n_señales":        len(señales),
        "n_operaciones":    len(validas),
        "wr_ops_pct":       round(wr_ops, 2),
        "pf_ops":           round(pf_ops, 3),
        "avg_pnl_pct":      round(avg_pnl, 3),
        "max_dd_ops_pct":   round(max_dd_ops, 2),
        "portfolio":        pf,
        "operaciones":      operaciones,
    }


def run_backtest(historico: dict, desde: datetime.date, hasta: datetime.date,
                 params: Params, estado_filtro: str = None,
                 n_workers: int = None) -> dict:
    """
    Punto de entrada único del motor.
    Recibe historico + params → devuelve métricas.
    Sin efectos secundarios — seguro para llamar en paralelo.
    """
    señales = construir_señales(historico, desde, hasta, params, n_workers)

    if estado_filtro:
        n_antes = len(señales)
        estados_en_señales = {s["estado"] for s in señales}
        señales = [s for s in señales
                   if estado_filtro.upper() in s["estado"].upper()]
        if not señales and n_antes > 0:
            return {"error": f"Sin señales {estado_filtro} (había {n_antes}; estados: {sorted(estados_en_señales)})",
                    "n_señales": 0, "params": params.as_dict()}

    if not señales:
        return {"error": "Sin señales", "n_señales": 0,
                "params": params.as_dict(), "periodo": f"{desde}→{hasta}"}

    # ── Simular operaciones en paralelo ──────────────────────────────────
    n_cores = n_workers or multiprocessing.cpu_count()
    _MIN_PARALLEL = 40  # overhead de spawn no compensa con pocas señales

    # A2/A3: dispatch sim via strategy registry (default "baseline" = original)
    try:
        from gpu_bt.strategies import get_strategy
        strategy_id = getattr(params, "strategy_id", "baseline") or "baseline"
        _, _sim_fn = get_strategy(strategy_id)
    except Exception:
        _sim_fn = simular_operacion

    if n_cores <= 1 or len(señales) < _MIN_PARALLEL:
        operaciones = [_sim_fn(s, historico, params) for s in señales]
    else:
        hkeys, hdata = serializar_historico(historico)
        params_dict  = params.as_dict()
        params_dict["excluir_tickers"] = list(params.excluir_tickers)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_cores,
                       initializer=_init_sim_worker,
                       initargs=(hkeys, hdata, params_dict)) as pool:
            operaciones = pool.map(_simular_operacion_worker, señales)

    pf = simular_portfolio(operaciones, params)

    validas = [o for o in operaciones if o["outcome"] != "SIN_DATOS"]
    wins    = [o for o in validas if "WIN" in o["outcome"]]
    losses  = [o for o in validas if o["outcome"] == "LOSS"]
    nv      = len(wins) + len(losses)
    pnls    = [o["pnl_pct"] for o in validas]
    pw      = [o["pnl_pct"] for o in wins]
    pl      = [o["pnl_pct"] for o in losses]

    wr_ops  = len(wins) / nv * 100 if nv else 0
    pf_ops  = sum(pw) / abs(sum(pl)) if pl and sum(pl) != 0 else 0
    avg_pnl = statistics.mean(pnls) if pnls else 0

    eq = peak = 100.0; max_dd_ops = 0.0
    for o in sorted(validas, key=lambda x: x["fecha"]):
        eq   *= 1 + o["pnl_pct"] / 100
        peak  = max(peak, eq)
        max_dd_ops = max(max_dd_ops, (peak - eq) / peak * 100)

    return {
        "params":           params.as_dict(),
        "params_label":     params.label(),
        "periodo":          f"{desde}→{hasta}",
        "n_señales":        len(señales),
        "n_operaciones":    len(validas),
        "wr_ops_pct":       round(wr_ops, 2),
        "pf_ops":           round(pf_ops, 3),
        "avg_pnl_pct":      round(avg_pnl, 3),
        "max_dd_ops_pct":   round(max_dd_ops, 2),
        "portfolio":        pf,
        "operaciones":      operaciones,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKERS PARA backtest_wf.py (deben estar en este módulo para spawn de Windows)
# ═══════════════════════════════════════════════════════════════════════════════

_WORKER_HISTORICO = None


def _worker_init(np_hist_bytes: bytes):
    """Pool initializer (modo fork/forkserver)."""
    global _WORKER_HISTORICO
    import pickle as _pk
    np_hist = _pk.loads(np_hist_bytes)
    _WORKER_HISTORICO = numpy_to_historico(np_hist)


def _worker_job(job: dict) -> dict:
    """
    Worker autónomo — carga el histórico desde disco.
    Compatible con spawn de Windows.
    """
    import time as _time
    t0 = _time.time()
    try:
        import datetime as _dt, pickle as _pk, sys as _sys, pathlib as _pl

        _root = str(_pl.Path(job.get("root_path", ".")).resolve())
        if _root not in _sys.path:
            _sys.path.insert(0, _root)

        from gpu_bt.core import (Params, run_backtest, numpy_to_historico,
                                   _deserializar_historico)

        params = Params(**{k: v for k, v in job["params_dict"].items()
                           if k in Params.__dataclass_fields__})

        global _WORKER_HISTORICO
        if _WORKER_HISTORICO is None:
            hist_path = job.get("hist_path")
            if hist_path and _pl.Path(hist_path).exists():
                with open(hist_path, "rb") as f:
                    np_hist = _pk.load(f)
                _WORKER_HISTORICO = numpy_to_historico(np_hist)
            elif "hkeys" in job:
                _WORKER_HISTORICO = _deserializar_historico(job["hkeys"], job["hdata"])
            else:
                raise RuntimeError(f"Sin histórico: hist_path={job.get('hist_path')}")

        def _todate(v):
            return v if isinstance(v, _dt.date) else _dt.date.fromisoformat(str(v)[:10])

        resultado = run_backtest(
            historico     = _WORKER_HISTORICO,
            desde         = _todate(job["desde"]),
            hasta         = _todate(job["hasta"]),
            params        = params,
            estado_filtro = job.get("estado_filtro"),
            n_workers     = 1,
        )

        pf = resultado.get("portfolio", {})
        return {
            "ventana_id":    job["ventana_id"],
            "modo":          job["modo"],
            "desde":         str(job["desde"]),
            "hasta":         str(job["hasta"]),
            "params_label":  params.label(),
            "params":        params.as_dict(),
            "n_señales":     resultado.get("n_señales", 0),
            "n_ops":         resultado.get("n_operaciones", 0),
            "wr_ops":        resultado.get("wr_ops_pct", 0),
            "pf_ops":        resultado.get("pf_ops", 0),
            "avg_pnl":       resultado.get("avg_pnl_pct", 0),
            "max_dd_ops":    resultado.get("max_dd_ops_pct", 0),
            "retorno_pct":   pf.get("retorno_pct", 0),
            "max_dd_pct":    pf.get("max_drawdown_pct", 0),
            "max_dd_dur_d":  pf.get("max_dd_duration_d", 0),
            "wr_pct":        pf.get("win_rate_pct", 0),
            "profit_factor": pf.get("profit_factor", 0),
            "sharpe":        pf.get("sharpe", 0),
            "sortino":       pf.get("sortino", 0),
            "calmar":        pf.get("calmar", 0),
            "expectancy_pct": pf.get("expectancy_pct", 0),
            "max_win_streak": pf.get("max_win_streak", 0),
            "max_loss_streak": pf.get("max_loss_streak", 0),
            "n_ejecutadas":  pf.get("n_ejecutadas", 0),
            "error":         resultado.get("error", ""),
            "elapsed_s":     round(_time.time() - t0, 1),
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "ventana_id":    job.get("ventana_id"),
            "modo":          job.get("modo"),
            "params_label":  str(job.get("params_dict", {}).get("rs_minimo", "?")),
            "error":         str(e),
            "traceback":     tb[:400],
            "retorno_pct":   0, "max_dd_pct": 100, "wr_pct": 0,
            "profit_factor": 0, "n_ops": 0, "n_ejecutadas": 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER TRAIN SWEEP — grid search paralelo para backtest_wf / backtest_capital
# ═══════════════════════════════════════════════════════════════════════════════
# El sweep de N params sobre candidatos pre-computados es CPU-bound puro Python.
# ThreadPoolExecutor quedaba GIL-bound (1 core real). ProcessPool con candidatos
# + histórico vía initializer (1 vez por worker, amortizado sobre los N params)
# da paralelismo real. run_backtest_rapido es determinista y sin side-effects →
# resultados IDÉNTICOS al modo serial, solo más rápido.

_TRAIN_CANDS  = None
_TRAIN_HIST   = None
_TRAIN_ESTADO = None


def _eval_train_params(params: "Params", candidatos: list,
                       historico: dict, estado_filtro: str = None) -> dict:
    """Evalúa un Params sobre candidatos pre-computados → dict métricas TRAIN.

    Pura, sin side-effects. Compartida por la rama serial y la paralela —
    misma función = mismo resultado garantizado.
    """
    r = run_backtest_rapido(candidatos, historico, params, estado_filtro)
    pf_d = r.get("portfolio", {})
    return {
        "params":        params.as_dict(),
        "params_label":  params.label(),
        "n_ops":         r.get("n_operaciones", 0),
        "retorno_pct":   pf_d.get("retorno_pct", 0) or 0,
        "max_dd_pct":    pf_d.get("max_drawdown_pct", 0) or 0,
        "wr_pct":        pf_d.get("win_rate_pct", 0) or 0,
        "profit_factor": pf_d.get("profit_factor", 0) or 0,
    }


def _init_train_worker(candidatos, hkeys, hdata, estado_filtro):
    """Pool initializer: candidatos + histórico deserializado 1 vez por worker."""
    global _TRAIN_CANDS, _TRAIN_HIST, _TRAIN_ESTADO
    _TRAIN_CANDS  = candidatos
    _TRAIN_HIST   = _deserializar_historico(hkeys, hdata)
    _TRAIN_ESTADO = estado_filtro


def _eval_train_worker(params: "Params") -> dict:
    """Worker map fn: usa los globals poblados por _init_train_worker."""
    return _eval_train_params(params, _TRAIN_CANDS, _TRAIN_HIST, _TRAIN_ESTADO)