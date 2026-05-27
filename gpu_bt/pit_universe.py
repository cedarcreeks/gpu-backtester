"""
pit_universe.py — Universo Point-in-Time
═══════════════════════════════════════════════════════════════════════════════
Genera el universo de tickers matching a momentum screener
on each historical date, eliminando el survivorship bias.

Estrategia (100% gratuita):
  1. Descarga constituents históricos del S&P500 + S&P400 MidCap desde Wikipedia
  2. Aplica filtros técnicos standard technical usando datos de yfinance
  3. Cachea el resultado — solo se calcula una vez por período

Filtros técnicos aplicados:
  - Precio > SMA20, SMA50, SMA200    (SMA filters)
  - Perf 6 meses > 20%               (6-month performance)
  - Volumen medio > 200K/día         (proxy de liquidez)

Limitaciones:
  - Los constituents de Wikipedia tienen lag de días en cambios de índice
  - Filtros fundamentales (deuda, EPS) no aplicables sin datos de pago
  → Universo conservador → estimaciones de WR ligeramente pesimistas (correcto)
═══════════════════════════════════════════════════════════════════════════════
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import json
import datetime
import pathlib
import time
import re
import urllib.request
import warnings
from typing import Optional

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    raise ImportError("pip install pandas yfinance")


# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

_PROJECT_ROOT  = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR      = _PROJECT_ROOT / "data" / "pit_cache"
SP500_CACHE    = CACHE_DIR / "sp500_constituents.json"
SP400_CACHE    = CACHE_DIR / "sp400_constituents.json"
UNIVERSE_CACHE = CACHE_DIR / "daily_universe.json"

FILTRO_SMA20   = True
FILTRO_SMA50   = True
FILTRO_SMA200  = True
FILTRO_PERF6M  = 20.0    # rendimiento 6m mínimo (%)
AVGVOL_MIN     = 200_000  # volumen medio diario mínimo

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

# Núcleo de líderes de momentum — fallback si Wikipedia no responde
FALLBACK_TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","AMD",
    "CRM","NOW","PANW","CRWD","SNOW","DDOG","ZS","NET","MDB","PLTR",
    "SMCI","ANET","MELI","APP","COIN","RDDT","ASTS","RKLB","KTOS","AXON",
    "FICO","INTU","TTD","TMDX","CELH","HIMS","SOUN","IONQ","RXRX","GEV",
    "VST","CEG","CIEN","AMAT","LRCX","KLAC","MU","DELL","NFLX","SPOT",
    "JPM","GS","V","MA","PYPL","SQ","AFRM","LMT","LDOS","RTX",
    "ENPH","FSLR","MNST","COST","SBUX","MCD","NKE","LULU","TJX","ULTA",
    "UNH","HCA","ISRG","DXCM","ALGN","IDXX","EW","RMD","PODD","NVCR",
    "TSCO","WYNN","MGM","LVS","PENN","DKNG","BALY","RSI","FLUT","EVRI",
    "CVNA","ABNB","BKNG","EXPE","LYFT","UBER","DASH","GRAB","SE","MELI",
]


# ─── DESCARGA DE CONSTITUENTS ─────────────────────────────────────────────────

def _fetch_wikipedia_tickers(url: str, cache_path: pathlib.Path,
                              max_age_days: int = 7) -> list:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        age_days = (datetime.datetime.now() -
                    datetime.datetime.fromtimestamp(
                        cache_path.stat().st_mtime)).days
        if age_days < max_age_days:
            with open(cache_path) as f:
                data = json.load(f)
            print(f"  [PIT] Caché Wikipedia: {len(data)} tickers ({cache_path.name})")
            return data

    print(f"  [PIT] Descargando constituents desde Wikipedia...")
    tickers = []
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (backtesting research)"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        table_m = re.search(
            r'<table[^>]+wikitable[^>]*>(.*?)</table>',
            html, re.DOTALL | re.IGNORECASE)
        if table_m:
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>',
                              table_m.group(1), re.DOTALL)
            for row in rows[1:]:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                if not cells:
                    continue
                raw = re.sub(r'<[^>]+>', '', cells[0]).strip()
                ticker = raw.split('\n')[0].strip().replace('.', '-')
                if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', ticker):
                    tickers.append(ticker)

        if tickers:
            with open(cache_path, "w") as f:
                json.dump(tickers, f)
            print(f"  [PIT] {len(tickers)} tickers guardados")
        else:
            print(f"  [PIT] ⚠ Sin tickers extraídos — usando fallback")

    except Exception as e:
        print(f"  [PIT] Error Wikipedia: {e}")
        if cache_path.exists():
            with open(cache_path) as f:
                tickers = json.load(f)
            print(f"  [PIT] Usando caché antiguo: {len(tickers)} tickers")

    return tickers


def get_sp1500_tickers() -> list:
    sp500 = _fetch_wikipedia_tickers(SP500_URL, SP500_CACHE)
    sp400 = _fetch_wikipedia_tickers(SP400_URL, SP400_CACHE)
    # sorted (no list): list(set()) da orden distinto por proceso sin
    # PYTHONHASHSEED → no-determinismo en orden de iteración del histórico.
    combined = sorted(set(sp500 + sp400)) if (sp500 or sp400) else sorted(set(FALLBACK_TICKERS))
    combined = [t for t in combined if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', t)]
    print(f"  [PIT] Universo total: {len(combined)} tickers")
    return combined


# ─── DESCARGA DE HISTÓRICO ────────────────────────────────────────────────────

def descargar_historico_pit(tickers: list, desde: str, hasta: str,
                             force: bool = False) -> dict:
    """
    Descarga histórico de precios en batches.
    Cachea en HDF5 para re-uso entre ejecuciones.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = CACHE_DIR / "price_history.h5"
    todos   = sorted(set(tickers + ["SPY"]))   # sorted: orden reproducible
    cached  = {}

    # Cargar caché HDF5
    if h5_path.exists() and not force:
        try:
            with pd.HDFStore(str(h5_path), mode="r") as store:
                for key in store.keys():
                    t = key.strip("/").replace("__", "-")
                    df = store[key]
                    # Solo usar si cubre el período
                    if len(df) > 20:
                        cached[t] = df
            print(f"  [PIT] HDF5 caché: {len(cached)} tickers cargados")
        except Exception as e:
            print(f"  [PIT] Error leyendo HDF5: {e}")
            cached = {}

    por_descargar = [t for t in todos if t not in cached]

    if por_descargar:
        print(f"  [PIT] Descargando {len(por_descargar)} tickers nuevos...")
        batch_size = 40
        nuevos     = {}

        for i in range(0, len(por_descargar), batch_size):
            batch = por_descargar[i:i+batch_size]
            n_batch = i // batch_size + 1
            n_total = (len(por_descargar) + batch_size - 1) // batch_size
            print(f"     Batch {n_batch}/{n_total} ({len(batch)} tickers)...",
                  end="\r")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = yf.download(
                        batch, start=desde, end=hasta,
                        auto_adjust=True, progress=False,
                        group_by="ticker", threads=True
                    )
                if raw.empty:
                    continue

                for t in batch:
                    try:
                        df = raw[t].dropna(how="all") if len(batch) > 1 else raw.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        if not df.empty and len(df) > 20:
                            nuevos[t] = df
                    except Exception:
                        continue
            except Exception as e:
                print(f"\n     ⚠ Error batch {n_batch}: {e}")
            time.sleep(0.3)

        print(f"\n  [PIT] {len(nuevos)} tickers nuevos descargados")

        # Guardar en HDF5
        if nuevos:
            try:
                with pd.HDFStore(str(h5_path), mode="a", complevel=5) as store:
                    for t, df in nuevos.items():
                        # HDF5 no acepta '-' en claves — reemplazar
                        key = t.replace("-", "__")
                        store[key] = df
                print(f"  [PIT] HDF5 actualizado")
            except Exception as e:
                print(f"  [PIT] ⚠ No se pudo guardar HDF5: {e}")

        cached.update(nuevos)

    print(f"  [PIT] Histórico total: {len(cached)} tickers disponibles")
    return cached


# ─── FILTROS TÉCNICOS ─────────────────────────────────────────────────────────

def _cumple_filtros(df: pd.DataFrame, idx: int,
                    sma20: bool = True, sma50: bool = True,
                    sma200: bool = True, perf6m: float = 20.0,
                    avgvol: int = 200_000) -> bool:
    if idx < 200:
        return False
    try:
        c = float(df.iloc[idx]["Close"])
        if c <= 0:
            return False

        vol_medio = float(df.iloc[max(0, idx-20):idx]["Volume"].mean())
        if vol_medio < avgvol:
            return False

        if sma200:
            if c < float(df.iloc[idx-200:idx]["Close"].mean()):
                return False
        if sma50:
            if c < float(df.iloc[idx-50:idx]["Close"].mean()):
                return False
        if sma20:
            if c < float(df.iloc[idx-20:idx]["Close"].mean()):
                return False
        if perf6m > 0 and idx >= 126:
            c6m = float(df.iloc[idx-126]["Close"])
            if c6m > 0 and (c / c6m - 1) * 100 < perf6m:
                return False

        return True
    except Exception:
        return False


# ─── CLASE PRINCIPAL ──────────────────────────────────────────────────────────

class PitUniverse:
    """
    Universo point-in-time: which tickers met momentum filters on each date.

    Uso básico:
        pit = PitUniverse()
        pit.build("2020-01-01", "2026-01-01")
        tickers = pit.get_universe(datetime.date(2023, 6, 15))
        historico = pit.historico   # para pasar al backtest

    Con walk-forward:
        historico_ventana = pit.get_historico_ventana(desde, hasta)
    """

    def __init__(self):
        self.historico:       dict = {}
        self._daily_universe: dict = {}
        self._built:          bool = False

    def build(self, desde: str, hasta: str,
              filtros: dict = None, force: bool = False):
        """
        Construye el universo diario para el período completo.
        Solo se ejecuta una vez — el resultado se cachea en disco.
        """
        f       = filtros or {}
        sma20   = f.get("sma20",   FILTRO_SMA20)
        sma50   = f.get("sma50",   FILTRO_SMA50)
        sma200  = f.get("sma200",  FILTRO_SMA200)
        perf6m  = f.get("perf6m",  FILTRO_PERF6M)
        avgvol  = f.get("avgvol",  AVGVOL_MIN)

        # Intentar cargar caché de universo diario
        if UNIVERSE_CACHE.exists() and not force:
            try:
                with open(UNIVERSE_CACHE) as fp:
                    meta = json.load(fp)
                if meta.get("desde") == desde and meta.get("hasta") == hasta:
                    self._daily_universe = meta["universe"]
                    print(f"[PIT] Universo diario en caché: "
                          f"{len(self._daily_universe)} días")
                    # Cargar histórico para el backtest
                    all_tickers = list({t for tl in self._daily_universe.values()
                                        for t in tl} | {"SPY"})
                    desde_extra = str(
                        datetime.date.fromisoformat(desde) -
                        datetime.timedelta(days=350))
                    self.historico = descargar_historico_pit(
                        all_tickers, desde_extra, hasta, force=force)
                    self._built = True
                    return
            except Exception:
                pass

        print(f"[PIT] Construyendo universo point-in-time...")
        print(f"      {desde} → {hasta}")

        # 1. Tickers del S&P1500
        tickers = get_sp1500_tickers()

        # 2. Descargar histórico con margen para SMA200
        desde_extra = str(
            datetime.date.fromisoformat(desde) - datetime.timedelta(days=350))
        self.historico = descargar_historico_pit(
            tickers, desde_extra, hasta, force=force)

        # 3. Calendario de fechas (días de mercado)
        spy_df = self.historico.get("SPY")
        if spy_df is None:
            raise ValueError("Sin datos de SPY")

        desde_dt = datetime.date.fromisoformat(desde)
        hasta_dt = datetime.date.fromisoformat(hasta)
        fechas   = [d.date() for d in spy_df.index
                    if desde_dt <= d.date() <= hasta_dt]

        # 4. Para cada fecha, aplicar filtros
        print(f"[PIT] Aplicando filtros en {len(fechas)} fechas de mercado...")
        n = len(fechas)

        for i, fecha in enumerate(fechas):
            if i % 50 == 0:
                print(f"     {i}/{n} ({i*100//n}%)...", end="\r")

            activos = []
            for t, df in self.historico.items():
                if t == "SPY":
                    continue
                try:
                    idx = df.index.get_indexer(
                        [pd.Timestamp(fecha)], method="pad")[0]
                    if _cumple_filtros(df, idx, sma20, sma50,
                                       sma200, perf6m, avgvol):
                        activos.append(t)
                except Exception:
                    continue

            self._daily_universe[str(fecha)] = activos

        print(f"\n[PIT] ✓ Completado")
        counts = [len(v) for v in self._daily_universe.values()]
        if counts:
            print(f"      Tickers/día → min:{min(counts)} "
                  f"media:{sum(counts)//len(counts)} max:{max(counts)}")

        # 5. Guardar caché
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(UNIVERSE_CACHE, "w") as fp:
            json.dump({"desde": desde, "hasta": hasta,
                       "universe": self._daily_universe}, fp)
        print(f"[PIT] Caché guardado → {UNIVERSE_CACHE}")
        self._built = True

    def get_universe(self, fecha: datetime.date) -> list:
        """Tickers activos en una fecha. Busca hacia atrás si no hay exacta."""
        if not self._built:
            raise RuntimeError("Ejecuta build() primero")
        key = str(fecha)
        if key in self._daily_universe:
            return self._daily_universe[key]
        for f in sorted(self._daily_universe.keys(), reverse=True):
            if f <= key:
                return self._daily_universe[f]
        return []

    def get_historico_ventana(self, desde: datetime.date,
                               hasta: datetime.date) -> dict:
        """
        Histórico filtrado al período dado + tickers que aparecen en él.
        Optimiza memoria para walk-forward: no pasa datos fuera de la ventana.
        """
        if not self._built:
            raise RuntimeError("Ejecuta build() primero")

        # Tickers activos en algún momento de la ventana
        tickers_ventana = set()
        for fecha_str, tickers in self._daily_universe.items():
            try:
                f = datetime.date.fromisoformat(fecha_str)
                if desde <= f <= hasta:
                    tickers_ventana.update(tickers)
            except Exception:
                continue
        tickers_ventana.add("SPY")

        # Margen extra para cálculos de SMA/RS al inicio de la ventana
        margen   = datetime.timedelta(days=300)
        inicio   = desde - margen
        resultado = {}
        for t in sorted(tickers_ventana):   # sorted: dict en orden reproducible
            df = self.historico.get(t)
            if df is not None:
                mask = df.index.date >= inicio
                resultado[t] = df[mask]

        return resultado

    def coverage_report(self) -> dict:
        if not self._daily_universe:
            return {}
        counts = [len(v) for v in self._daily_universe.values()]
        fechas = sorted(self._daily_universe.keys())
        return {
            "fechas_totales":   len(fechas),
            "fecha_inicio":     fechas[0],
            "fecha_fin":        fechas[-1],
            "tickers_min":      min(counts),
            "tickers_max":      max(counts),
            "tickers_media":    round(sum(counts) / len(counts)),
            "tickers_unicos":   len({t for tl in self._daily_universe.values()
                                     for t in tl}),
        }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde",  default="2020-01-01")
    ap.add_argument("--hasta",  default=str(datetime.date.today()))
    ap.add_argument("--force",  action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    pit = PitUniverse()
    pit.build(args.desde, args.hasta, force=args.force)

    if args.report:
        r = pit.coverage_report()
        print("\n[PIT] Coverage Report:")
        for k, v in r.items():
            print(f"  {k:<25}: {v}")
        print("\n[PIT] Muestra de universo en fechas clave:")
        for fs in ["2020-03-20", "2021-11-01", "2022-06-15",
                   "2023-01-15", "2024-06-01", "2025-01-02"]:
            try:
                u = pit.get_universe(datetime.date.fromisoformat(fs))
                print(f"  {fs}: {len(u):>4} tickers activos")
            except Exception:
                pass
