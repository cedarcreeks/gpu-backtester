"""
bt_cache.py — Caché Parquet + Numba para el sistema de backtesting
═══════════════════════════════════════════════════════════════════════════════
Mejoras de rendimiento sin tocar la lógica de analyzer.py:

  1. CACHÉ PARQUET
     - descargar_o_cargar()  → descarga una sola vez, guarda en disco
     - La segunda vez lee del .parquet local (10-20× más rápido que yfinance)
     - Refresca automáticamente si los datos tienen más de N días de antigüedad

  2. NUMBA JIT
     - _calcular_perfs_nb()  → calcula performances (1W/1M/3M/6M/1Y) sobre
       arrays numpy en velocidad C, sin loops Python por cada celda
     - _calcular_rs_nb()     → percentiles ponderados vectorizados
     - _simular_op_nb()      → simulación de stop/T1/T2 en C puro
     - Todo es compatible con multiprocessing (las funciones JIT son globales)

Uso:
    from gpu_bt.cache import descargar_o_cargar, preparar_arrays_nb

    historico = descargar_o_cargar(tickers, desde_str, hasta_str)
    np_hist   = preparar_arrays_nb(historico)   # para las funciones Numba

Dependencias:
    pip install yfinance pandas pyarrow numba numpy
═══════════════════════════════════════════════════════════════════════════════
"""

import datetime
import pathlib
import warnings

try:
    import numpy as np
    import pandas as pd
except ImportError:
    raise ImportError("pip install pandas numpy pyarrow")

# ── Numba: opcional, con fallback puro-Python ──────────────────────────────
try:
    from numba import njit, prange
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    # Decoradores noop para que el código funcione sin Numba
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator
    def prange(n):
        return range(n)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = _PROJECT_ROOT / ".bt_cache"


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CACHÉ PARQUET
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_path(ticker: str) -> pathlib.Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{ticker}.parquet"


def _necesita_refresco(ticker: str, hasta: datetime.date, max_age_days: int = 1) -> bool:
    """True si el archivo no existe o es más viejo que max_age_days."""
    p = _cache_path(ticker)
    if not p.exists():
        return True
    mtime = datetime.date.fromtimestamp(p.stat().st_mtime)
    # Si 'hasta' es el pasado (no hoy), los datos son estáticos → nunca refrescar
    if hasta < datetime.date.today() - datetime.timedelta(days=5):
        return False
    return (datetime.date.today() - mtime).days >= max_age_days


def _guardar_parquet(ticker: str, df: pd.DataFrame):
    try:
        df_save = df.copy()
        if df_save.index.tz is not None:
            df_save.index = df_save.index.tz_convert(None)
        df_save.to_parquet(_cache_path(ticker), engine="pyarrow", compression="snappy")
    except Exception as e:
        pass  # Si falla el guardado no es crítico


def _leer_parquet(ticker: str, desde_str: str, hasta_str: str) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(_cache_path(ticker), engine="pyarrow")
        # Filtrar al rango pedido (el parquet puede tener más datos)
        desde = pd.Timestamp(desde_str)
        hasta = pd.Timestamp(hasta_str)
        return df[(df.index >= desde) & (df.index <= hasta)]
    except Exception:
        return None


def descargar_o_cargar(tickers: list,
                       desde_str: str,
                       hasta_str: str,
                       sector_etfs: list = None,
                       max_age_days: int = 1,
                       batch_size: int = 50,
                       verbose: bool = True) -> dict:
    """
    Descarga tickers con yfinance y los guarda en caché Parquet.
    Las siguientes ejecuciones cargan desde disco (10-20× más rápido).

    Args:
        tickers:      lista de símbolos a descargar
        desde_str:    fecha inicio en formato "YYYY-MM-DD" (incluye warmup)
        hasta_str:    fecha fin
        sector_etfs:  ETFs de sector (se guardan con prefijo "ETF_")
        max_age_days: días antes de refrescar datos de mercado recientes
        batch_size:   tickers por llamada a yfinance
        verbose:      imprimir progreso

    Returns:
        dict {ticker: pd.DataFrame, "ETF_XLK": pd.DataFrame, ...}
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    sector_etfs = sector_etfs or []
    hasta       = datetime.date.fromisoformat(hasta_str)

    # sorted (no list) — list(set()) da orden distinto por proceso sin
    # PYTHONHASHSEED → orden de iteración del histórico no-determinista.
    # Defensa raíz del bug de reproducibilidad WF (ver backtest_core sorts).
    todos    = sorted(set(tickers + ["SPY"] + sector_etfs))
    historico = {}
    pendientes = []   # tickers que hay que descargar
    cargados   = 0

    # ── Cargar desde caché ────────────────────────────────────────────────────
    for t in todos:
        if not _necesita_refresco(t, hasta, max_age_days):
            df = _leer_parquet(t, desde_str, hasta_str)
            if df is not None and len(df) > 20:
                key = f"ETF_{t}" if t in sector_etfs else t
                historico[key] = df
                cargados += 1
                continue
        pendientes.append(t)

    if verbose and cargados:
        print(f"   📂 {cargados} tickers desde caché Parquet")
    if verbose and pendientes:
        print(f"   📥 {len(pendientes)} tickers a descargar (nuevos o desactualizados)...")

    # ── Descargar los que faltan ──────────────────────────────────────────────
    if pendientes:
        for i in range(0, len(pendientes), batch_size):
            batch = pendientes[i:i + batch_size]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = yf.download(
                        batch,
                        start=desde_str,
                        end=hasta_str,
                        auto_adjust=True,
                        progress=False,
                        group_by="ticker",
                        threads=True,
                    )
                if raw.empty:
                    continue
                for t in batch:
                    try:
                        df = raw[t].dropna(how="all") if len(batch) > 1 else raw.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        if df.empty or len(df) < 20:
                            continue
                        _guardar_parquet(t, df)
                        key = f"ETF_{t}" if t in sector_etfs else t
                        # Filtrar al rango desde–hasta para consistencia
                        mask = (df.index >= pd.Timestamp(desde_str)) & \
                               (df.index <= pd.Timestamp(hasta_str))
                        historico[key] = df[mask]
                    except Exception:
                        continue
            except Exception as e:
                if verbose:
                    print(f"   ⚠ Error en batch {i//batch_size + 1}: {e}")

            if verbose and len(pendientes) > batch_size:
                done = min(i + batch_size, len(pendientes))
                print(f"   {done}/{len(pendientes)} descargados...")

            import time; time.sleep(0.2)   # rate-limit suave

    if verbose:
        print(f"   ✓ Histórico listo: {len(historico)} tickers "
              f"({cargados} caché, {len(historico)-cargados} descargados)")

    return historico


def limpiar_cache(dias: int = 30):
    """Elimina archivos Parquet más viejos que `dias` días."""
    if not CACHE_DIR.exists():
        return
    hoy = datetime.date.today()
    eliminados = 0
    for p in CACHE_DIR.glob("*.parquet"):
        mtime = datetime.date.fromtimestamp(p.stat().st_mtime)
        if (hoy - mtime).days > dias:
            p.unlink()
            eliminados += 1
    print(f"   🗑  {eliminados} archivos eliminados del caché")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. NUMBA — funciones críticas del loop de escaneo
# ═══════════════════════════════════════════════════════════════════════════════
#
# Estrategia: el loop externo (fechas × tickers) se mantiene en Python
# porque necesita indexación dinámica y dicts.  Las operaciones NUMÉRICAS
# dentro del loop (calcular performances, RS ranks, stops, simulación) se
# compilan a C con Numba.
#
# Columnas del array numpy por ticker: [Open=0, High=1, Low=2, Close=3, Volume=4]

COL_OPEN  = 0
COL_HIGH  = 1
COL_LOW   = 2
COL_CLOSE = 3
COL_VOL   = 4


@njit(cache=True)
def calcular_perfs_nb(closes: np.ndarray, i: int) -> tuple:
    """
    Calcula los 5 retornos de performance desde el índice i.
    Returns: (perf1w, perf1m, perf3m, perf6m, perf1y)
    Devuelve 0.0 si no hay suficientes datos.
    """
    c0 = closes[i]
    if c0 <= 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    def _ret(ventana):
        j = i - ventana
        if j < 0:
            return 0.0
        cn = closes[j]
        return (c0 / cn - 1.0) * 100.0 if cn > 0.0 else 0.0

    return (_ret(5), _ret(21), _ret(63), _ret(126), _ret(252))


@njit(cache=True)
def calcular_rs_scores_nb(perfs_matrix: np.ndarray,
                           pesos: np.ndarray) -> np.ndarray:
    """
    Calcula RS Rating para un universo de n tickers.

    Args:
        perfs_matrix: array (n, 5) con [perf1w, perf1m, perf3m, perf6m, perf1y]
        pesos:        array (5,) con [0.10, 0.25, 0.25, 0.20, 0.20]

    Returns:
        rs_ratings: array (n,) con valores 1-99
    """
    n = perfs_matrix.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    scores = np.zeros(n, dtype=np.float64)

    # Para cada período: calcular percentil rank de cada ticker
    for p in range(5):
        vals = perfs_matrix[:, p]
        # Argsort estable (bubble sort simple — n típico < 200)
        orden = np.argsort(vals)
        for rank in range(n):
            orig_i = orden[rank]
            pct = (rank / (n - 1)) * 100.0 if n > 1 else 50.0
            scores[orig_i] += pct * pesos[p]

    # Normalizar a 1-99
    s_min = scores[0]
    s_max = scores[0]
    for k in range(1, n):
        if scores[k] < s_min:
            s_min = scores[k]
        if scores[k] > s_max:
            s_max = scores[k]

    rs = np.empty(n, dtype=np.float64)
    rango = s_max - s_min
    for k in range(n):
        if rango > 0.0:
            rs[k] = ((scores[k] - s_min) / rango) * 98.0 + 1.0
        else:
            rs[k] = 50.0

    return rs


@njit(cache=True)
def calcular_vol_rel_nb(volumen: np.ndarray, i: int) -> float:
    """Volumen relativo: vol_hoy / media_20d."""
    vol_hoy = volumen[i]
    start   = i - 20 if i >= 20 else 0
    if start >= i:
        return 1.0
    total = 0.0
    count = 0
    for k in range(start, i):
        total += volumen[k]
        count += 1
    if count == 0 or total == 0.0:
        return 1.0
    return vol_hoy / (total / count)


@njit(cache=True)
def calcular_vol_avg_nb(volumen: np.ndarray, i: int, ventana: int = 20) -> float:
    """Media de volumen en los últimos `ventana` días."""
    start = i - ventana if i >= ventana else 0
    total = 0.0
    count = 0
    for k in range(start, i):
        total += volumen[k]
        count += 1
    return total / count if count > 0 else 0.0


@njit(cache=True)
def calcular_h52_l52_nb(highs: np.ndarray, lows: np.ndarray, i: int) -> tuple:
    """Máximo y mínimo de las últimas 252 barras hasta i (inclusive)."""
    start = i - 252 if i >= 252 else 0
    h52 = highs[start]
    l52 = lows[start]
    for k in range(start + 1, i + 1):
        if highs[k] > h52:
            h52 = highs[k]
        if lows[k] < l52:
            l52 = lows[k]
    return h52, l52


@njit(cache=True)
def calcular_vol_sem_nb(closes: np.ndarray, i: int) -> float:
    """Volatilidad semanal media (para régimen de vol)."""
    total = 0.0
    count = 0
    max_w = min(12, i // 5)
    for w in range(max_w):
        p0 = closes[i - (w + 1) * 5]
        p1 = closes[i - w * 5]
        if p0 > 0.0:
            ret = p1 / p0 - 1.0
            if ret < 0.0:
                ret = -ret
            total += ret * 100.0
            count += 1
    return total / count if count > 0 else 3.0


@njit(cache=True)
def calcular_calidad_vela_nb(open_: float, high: float, low: float,
                              close: float, rel_vol: float) -> float:
    """Score de calidad de vela OHLC (0-100)."""
    rango  = high - low
    if rango <= 0.0:
        return 50.0
    cuerpo      = close - open_
    if cuerpo < 0.0:
        cuerpo = -cuerpo
    ratio_cuerpo = cuerpo / rango
    pos_cierre   = (close - low) / rango
    alcista      = 1.0 if close > open_ else 0.0
    rv_clamped   = rel_vol if rel_vol < 3.0 else 3.0
    return (ratio_cuerpo * 40.0 +
            pos_cierre   * 30.0 +
            alcista      * 15.0 +
            (rv_clamped / 3.0) * 15.0)


@njit(cache=True)
def simular_operacion_nb(opens: np.ndarray,
                          highs: np.ndarray,
                          lows: np.ndarray,
                          closes: np.ndarray,
                          spy_closes: np.ndarray,
                          spy_sma200: np.ndarray,
                          idx0: int,
                          entrada: float,
                          stop: float,
                          t1: float,
                          t2: float,
                          max_hold: int,
                          filtro_mercado: bool,
                          fraccion_t1: float,
                          slippage_pct: float,
                          trailing_stop_pct: float = 0.07) -> tuple:
    """
    Simula el ciclo de vida de una operación sobre arrays numpy.

    Returns: (pnl_pct, outcome_code, dias, precio_salida, motivo_code, t1_hit)
      outcome_code: 0=LOSS, 1=WIN_T1, 2=WIN_T2, 3=WIN_PARCIAL, 4=BREAKEVEN, 5=SIN_DATOS
      motivo_code:  0=stop, 1=t2, 2=filtro_mercado, 3=expirado, 4=fin_datos
    """
    entrada_eff = entrada * (1.0 + slippage_pct)
    stop_actual   = stop
    t1_hit        = False
    fraccion_viva = 1.0
    pnl_parcial   = 0.0
    precio_salida = entrada_eff
    motivo        = 3   # expirado
    dias          = 0
    n             = len(closes)
    max_precio    = entrada_eff

    for j in range(1, max_hold + 1):
        idx = idx0 + j
        if idx >= n:
            precio_salida = closes[n - 1]
            motivo = 4; dias = j; break

        op_ = opens[idx]
        hi_ = highs[idx]
        lo_ = lows[idx]
        cl_ = closes[idx]
        dias = j

        # Filtro mercado
        if filtro_mercado and not t1_hit and idx < len(spy_closes):
            if spy_closes[idx] < spy_sma200[idx]:
                precio_salida = op_; motivo = 2; break

        # Stop
        if lo_ <= stop_actual:
            precio_salida = op_ if op_ < stop_actual else stop_actual
            motivo = 0; break

        # T1
        if not t1_hit and hi_ >= t1:
            t1_hit        = True
            fraccion_viva = 1.0 - fraccion_t1
            pnl_parcial   = (t1 - entrada_eff) / entrada_eff * fraccion_t1
            if stop_actual < entrada_eff:
                stop_actual = entrada_eff
            max_precio = hi_
            if hi_ >= t2:
                precio_salida = t2; motivo = 1; break
            continue

        # Trailing stop post-T1
        if t1_hit and trailing_stop_pct > 0.0:
            if hi_ > max_precio:
                max_precio = hi_
            trail = max_precio * (1.0 - trailing_stop_pct)
            if trail > stop_actual:
                stop_actual = trail

        # T2
        if t1_hit and hi_ >= t2:
            precio_salida = t2; motivo = 1; break

        if j == max_hold:
            precio_salida = cl_; motivo = 3

    pnl_total = pnl_parcial + (precio_salida - entrada_eff) / entrada_eff * fraccion_viva
    pnl_pct   = pnl_total * 100.0

    if motivo == 0:
        outcome = 0       # LOSS
    elif motivo == 1:
        outcome = 2       # WIN_T2
    elif t1_hit:
        outcome = 1       # WIN_T1
    elif pnl_pct > 0.5:
        outcome = 3       # WIN_PARCIAL
    elif pnl_pct < -0.5:
        outcome = 0       # LOSS
    else:
        outcome = 4       # BREAKEVEN

    return (pnl_pct, outcome, dias, precio_salida, motivo, t1_hit)


# ── Constantes de outcome/motivo para el código Python ───────────────────────
OUTCOME_NAMES = {0: "LOSS", 1: "WIN_T1", 2: "WIN_T2",
                 3: "WIN_PARCIAL", 4: "BREAKEVEN", 5: "SIN_DATOS"}
MOTIVO_NAMES  = {0: "stop", 1: "t2", 2: "filtro_mercado",
                 3: "expirado", 4: "fin_datos"}

# Pesos RS en el orden que usa calcular_rs_scores_nb: 1W, 1M, 3M, 6M, 1Y
_PESOS_NB = np.array([0.10, 0.25, 0.25, 0.20, 0.20], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPARACIÓN DE ARRAYS NUMPY
# ═══════════════════════════════════════════════════════════════════════════════

def preparar_arrays_nb(historico: dict) -> dict:
    """
    Convierte el histórico de DataFrames a arrays numpy float64 contiguos.
    Mucho más rápido de acceder en el loop de escaneo.

    Returns:
        dict {ticker: {"opens", "highs", "lows", "closes", "volumes", "dates"}}
    """
    result = {}
    for t, df in historico.items():
        try:
            arr = df[["Open","High","Low","Close","Volume"]].to_numpy(dtype=np.float64, na_value=0.0)
            dates = np.array([str(d.date()) for d in df.index], dtype=object)
            result[t] = {
                "opens":   arr[:, 0],
                "highs":   arr[:, 1],
                "lows":    arr[:, 2],
                "closes":  arr[:, 3],
                "volumes": arr[:, 4],
                "dates":   dates,
            }
        except Exception:
            continue
    return result


def get_idx_nb(arrays: dict, ticker: str, ts_str: str) -> int:
    """
    Busca el índice del array más cercano (≤) a la fecha ts_str.
    ts_str formato "YYYY-MM-DD".
    Devuelve -1 si no se encuentra.
    """
    dates = arrays[ticker]["dates"]
    # Búsqueda binaria manual (las fechas están ordenadas)
    lo, hi = 0, len(dates) - 1
    result = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if dates[mid] <= ts_str:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def warmup_numba():
    """
    Fuerza la compilación JIT de las funciones Numba con datos pequeños.
    Llamar UNA VEZ antes del bucle principal para que el primer backtest
    no pague el coste de compilación (~2-5s).
    """
    if not _NUMBA_OK:
        return
    print("   [*] Compilando funciones Numba (solo la primera vez)...", end=" ", flush=True)
    dummy_closes = np.ones(300, dtype=np.float64) * 100.0
    dummy_vol    = np.ones(300, dtype=np.float64) * 1_000_000.0
    dummy_hi     = dummy_closes * 1.01
    dummy_lo     = dummy_closes * 0.99
    dummy_op     = dummy_closes.copy()
    dummy_perfs  = np.random.randn(10, 5)

    calcular_perfs_nb(dummy_closes, 260)
    calcular_rs_scores_nb(dummy_perfs, _PESOS_NB)
    calcular_vol_rel_nb(dummy_vol, 250)
    calcular_vol_avg_nb(dummy_vol, 250)
    calcular_h52_l52_nb(dummy_hi, dummy_lo, 260)
    calcular_vol_sem_nb(dummy_closes, 260)
    calcular_calidad_vela_nb(100.0, 102.0, 98.0, 101.0, 1.5)
    simular_operacion_nb(
        dummy_op, dummy_hi, dummy_lo, dummy_closes,
        dummy_closes, dummy_closes * 0.99,
        250, 100.0, 93.0, 114.0, 121.0,
        20, True, 0.5, 0.001, 0.05
    )
    print("✓")


def tiene_numba() -> bool:
    return _NUMBA_OK
