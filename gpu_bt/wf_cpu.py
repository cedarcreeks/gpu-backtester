"""
backtest_wf.py — Walk-Forward Backtesting Engine, fiel a analyzer.py
═══════════════════════════════════════════════════════════════════════════════
Valida el sistema de momentum con rigor profesional usando la lógica EXACTA
de analyzer.py (RS Rating 5 períodos, clasificar_estado, calcular_posicion).

  WALK-FORWARD: evita overfitting
  ─────────────────────────────────────────────────────────────────────────────
  El período completo se divide en ventanas rodantes:
    - Train: busca la combinación óptima de parámetros (grid search)
    - Test:  evalúa esa combinación en datos nunca vistos

  ARQUITECTURA PRE-COMPUTE (rápida)
  ─────────────────────────────────────────────────────────────────────────────
  Por cada ventana:
    1. precomputar_candidatos() — escanea histórico UNA sola vez (~10s)
    2. Para cada params del grid → filtrar_candidatos() + simular (~ms)

  Complejidad: O(ventanas × tickers × fechas) vs O(ventanas × params × ...)
  Con 4 ventanas y 96 params: 96× más rápido que el enfoque ingenuo.

  GRID DE PARÁMETROS — alineado con settings.py
  ─────────────────────────────────────────────────────────────────────────────
  Explora variaciones de RS mínimo, score, rvol, calidad de vela y filtros,
  con los PESOS RS fijos (1Y=20%, 6M=20%, 3M=25%, 1M=25%, 1W=10%) porque
  representan el sistema real en producción.

Uso:
  pip install yfinance pandas numpy

  python backtest_wf.py                        # universo fijo, params por defecto
  python backtest_wf.py --desde 2019-01-01
  python backtest_wf.py --ventanas 6
  python backtest_wf.py --grid rapido          # más rápido (~2 min)
  python backtest_wf.py --estado BREAKOUT
  python backtest_wf.py --guardar
═══════════════════════════════════════════════════════════════════════════════
"""
import sys as _sys, os as _os

# Forzar UTF-8 en Windows (cp1252 no soporta checkmarks/emojis)
if _sys.stdout and hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import sys
import json
import datetime
import pathlib
import argparse
import time
import multiprocessing
import statistics
import itertools

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from gpu_bt.core import (
    Params, run_backtest, run_backtest_rapido,
    precomputar_candidatos, serializar_historico,
    _eval_train_params, _init_train_worker, _eval_train_worker,
)

# Caché Parquet + Numba
try:
    from gpu_bt.cache import descargar_o_cargar as _cache_descargar, warmup_numba, tiene_numba
    _CACHE_OK = True
except ImportError:
    _CACHE_OK = False

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = _PROJECT_ROOT / "reports"

# ETFs de sector — se guardan con prefijo "ETF_" en el histórico
# para que backtest_core los distinga de tickers operables
SECTOR_ETFS = ["XLK","XLV","XLF","XLY","XLP","XLI","XLE","XLB","XLRE","XLU","XLC"]


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIVERSO FIJO (fallback sin PIT)
# ═══════════════════════════════════════════════════════════════════════════════

UNIVERSO_FIJO = [
    # Semiconductores
    "NVDA","AMD","AVGO","QCOM","MRVL","SMCI","AMAT","LRCX","KLAC","ONTO",
    "ENTG","ACLS","MTSI","AMBA","MU","ADI","NXPI","TXN","MPWR","ON",
    # Software / Cloud / SaaS
    "CRM","NOW","PANW","CRWD","SNOW","DDOG","ZS","NET","MDB","GTLB",
    "HUBS","BILL","WDAY","VEEV","TEAM","ADBE","INTU","CDNS","SNPS","PCTY",
    # IA / Data / Infra
    "PLTR","APP","ANET","ORCL","PSTG","VRT","DELL","AXON",
    # Biotech / Medtech
    "ISRG","DXCM","PODD","GEHC","INSP","TMDX","EXAS","NTRA","VRTX","REGN",
    "ALGN","HOLX","ICLR","MEDP","ALNY","HALO","RARE","FOLD","SRPT",
    # Industrial / Defensa
    "GEV","PWR","EME","CBRE","TTEK","MTZ","IBP","BLDR","MLM","VMC","NUE",
    "KTOS","RKLB","LMT","RTX","NOC","GD","HWM","LDOS","BAH","CACI","SAIC",
    # Energía / Utilities
    "VST","CEG","NRG","OKE","LNG","CTRA","FANG","DVN","MRO","AR","EQT",
    # Financiero / Cripto
    "COIN","HOOD","IBKR","LPLA","GS","MS","JPM","BX","KKR","APO","ARES",
    # Consumo / Ocio
    "LULU","DECK","ONON","CELH","WING","BROS","DKNG","CVNA","TSLA",
    "RCL","HLT","MAR","BKNG","ABNB","UBER",
    # Salud / Farma
    "ELV","MOH","UNH","LLY","NVO","ABBV","MRK",
    # Medios / Ad-tech
    "META","SNAP","PINS","RDDT","NFLX","SPOT",
    # REIT
    "CSGP","AMT","EQIX","PLD","EXR",
    # CPG / Bebidas
    "MNST","STZ","FIZZ","CELH",
    # Mega momentum
    "SHOP","MELI","SE","RBLX","DUOL","RDDT",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  GRID DE PARÁMETROS
#  Explora las dimensiones relevantes del sistema real (analyzer.py).
#  Los pesos RS NO se exploran — son fijos (1Y=20%,6M=20%,3M=25%,1M=25%,1W=10%).
# ═══════════════════════════════════════════════════════════════════════════════

def build_param_grid(modo: str = "normal", strategy_id: str = "baseline",
                     strategy_params: dict = None) -> list:
    """
    Genera el grid de combinaciones de Params a explorar.

    Dimensiones exploradas:
      - rs_minimo:        RS mínimo del universo
      - score_minimo:     score de confianza mínimo
      - rvol_minimo:      volumen relativo mínimo para BREAKOUT
      - calidad_vela_min: score mínimo de calidad de vela
      - filtro_earnings:  excluir señales con gap earnings en 5d
      - filtro_mercado:   solo operar si SPY>SMA200

    Modos:
      "rapido"   — ~16 combinaciones, < 3 min por ventana
      "normal"   — ~48 combinaciones, ~8 min por ventana
      "completo" — ~128 combinaciones, ~20 min por ventana
    """
    if modo == "rapido":
        rs_mins    = [50, 70, 85]
        scores     = [0.0, 55.0]
        rvols      = [1.0, 1.5]
        velas      = [0.0, 40.0]
        earnings   = [True]
        mercados   = [True]
        trailings  = [0.0, 0.07]
    elif modo == "normal":
        rs_mins    = [50, 70, 80, 85]
        scores     = [0.0, 50.0, 60.0]
        rvols      = [1.0, 1.5, 2.0]
        velas      = [0.0, 35.0, 55.0]
        earnings   = [True, False]
        mercados   = [True]
        trailings  = [0.0, 0.05, 0.07]
    else:  # completo
        rs_mins    = [50, 60, 70, 80, 85, 90]
        scores     = [0.0, 45.0, 55.0, 65.0, 70.0]
        rvols      = [1.0, 1.5, 2.0]
        velas      = [0.0, 30.0, 45.0, 60.0]
        earnings   = [True, False]
        mercados   = [True, False]
        trailings  = [0.0, 0.03, 0.05, 0.07]

    sp = strategy_params or {}

    # adx_trend / all_filters: explorar el umbral ADX como dimensión del grid.
    # Sin esto la WF testeaba adx_min fijo (default 25) — y 25 sobre-filtraba
    # el grind de baja-ADX (ver V3 del A/B 2026-05-14). Que la WF-TRAIN tunee
    # el umbral por ventana. {20,25,30} = leve/estándar/fuerte (Wilder).
    if strategy_id in ("adx_trend", "all_filters", "b3_b7"):
        if modo == "completo":
            adx_mins = [20.0, 25.0, 30.0]
        else:
            adx_mins = [20.0, 25.0]
    else:
        adx_mins = [None]   # sin dimensión extra para otras strategies

    # vol_atr: explorar atr_mult — multiplicador define la distancia del stop.
    # Tras WF A/B 2026-05-15 (5/6 fail max_dd_pp +6.47pp): atr_mult=1.5 inflaba
    # qty en vol baja → sobre-apalancamiento → DD spike asimétrico. Drop 1.5
    # del grid rapido/normal. Conservar en completo para investigación.
    if strategy_id == "vol_atr":
        if modo == "completo":
            atr_mults = [1.5, 2.0, 2.5, 3.0]
        else:
            atr_mults = [2.0, 2.5]
    else:
        atr_mults = [None]

    # B10 heat_cap: cap del riesgo agregado abierto. heat_cap=1.0 (sin cap)
    # incluido para que la WF compare con baseline behavior. Con max_pos=5 y
    # riesgo_por_op=0.02 el heat máximo natural es 10% → caps menores (5%, 8%)
    # fuerzan menos posiciones simultáneas en regimen de full risk.
    if strategy_id in ("heat_cap", "all_filters"):
        if modo == "completo":
            heat_caps = [0.04, 0.06, 0.08, 0.10, 1.0]
        else:
            heat_caps = [0.06, 0.08, 0.10]
    else:
        heat_caps = [None]

    grid = []
    for rs, sc, rv, vl, ea, mk, tr in itertools.product(
            rs_mins, scores, rvols, velas, earnings, mercados, trailings):
        for adx_min in adx_mins:
            for atr_mult in atr_mults:
                for hc in heat_caps:
                    _sp = dict(sp)
                    if adx_min is not None:
                        _sp["adx_min"] = adx_min
                    if atr_mult is not None:
                        _sp["atr_mult"] = atr_mult
                    p_kwargs = dict(
                        rs_minimo          = rs,
                        score_minimo       = sc,
                        rvol_minimo        = rv,
                        calidad_vela_min   = vl,
                        filtro_earnings    = ea,
                        filtro_mercado     = mk,
                        trailing_stop_pct  = tr,
                        strategy_id        = strategy_id,
                        strategy_params    = _sp,
                    )
                    if hc is not None:
                        p_kwargs["heat_cap"] = hc
                    grid.append(Params(**p_kwargs))

    # Deduplicar por label
    seen, unique = set(), []
    for p in grid:
        lbl = p.label()
        if lbl not in seen:
            seen.add(lbl); unique.append(p)

    return unique


# ═══════════════════════════════════════════════════════════════════════════════
#  VENTANAS WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════

def build_ventanas(desde: datetime.date, hasta: datetime.date,
                   n_ventanas: int = 5,
                   ratio_train: float = 0.7) -> list:
    """
    Divide el período en ventanas rodantes train/test.

    Returns lista de dicts:
        {ventana, train_desde, train_hasta, test_desde, test_hasta}
    """
    total_dias = (hasta - desde).days
    paso       = total_dias // n_ventanas
    ventana_d  = paso + int(paso * (1 / (1 - ratio_train) - 1))

    ventanas = []
    for i in range(n_ventanas):
        v_inicio   = desde + datetime.timedelta(days=i * paso)
        v_fin      = min(v_inicio + datetime.timedelta(days=ventana_d), hasta)
        train_dias = int((v_fin - v_inicio).days * ratio_train)
        train_fin  = v_inicio + datetime.timedelta(days=train_dias)
        test_inicio = train_fin + datetime.timedelta(days=1)

        if test_inicio >= v_fin:
            break
        if (v_fin - test_inicio).days < 60:  # test mínimo 60 días
            break

        ventanas.append({
            "ventana":     i + 1,
            "train_desde": v_inicio,
            "train_hasta": train_fin,
            "test_desde":  test_inicio,
            "test_hasta":  v_fin,
        })

    return ventanas


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORING DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════

def _score_resultado(r: dict) -> float:
    """Función de ranking para seleccionar el mejor params en train."""
    pf  = r.get("profit_factor", 0) or 0
    wr  = r.get("wr_pct", 0) or 0
    dd  = r.get("max_dd_pct", 100) or 100
    n   = r.get("n_ops", 0) or 0
    ret = r.get("retorno_pct", 0) or 0
    if n < 3:
        return -999.0
    if dd > 50:
        return -dd
    return round((ret / max(dd, 1)) * (1 + wr / 100) * (1 + min(pf, 5) / 10), 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run_walk_forward(historico: dict,
                     desde: datetime.date,
                     hasta: datetime.date,
                     n_ventanas: int    = 5,
                     ratio_train: float = 0.7,
                     param_grid: list   = None,
                     estado_filtro: str = None,
                     n_workers: int     = None,
                     verbose: bool      = True) -> dict:
    """
    Walk-forward con arquitectura pre-compute.

    Por cada ventana:
      1. precomputar_candidatos() — escanea historico UNA sola vez
      2. Para cada params del grid → filtrar_candidatos() + simular (ms)
      3. Seleccionar mejor params (train) → evaluar en test (datos nunca vistos)
    """
    n_cpu    = n_workers or multiprocessing.cpu_count()
    grid     = param_grid or build_param_grid("normal")
    ventanas = build_ventanas(desde, hasta, n_ventanas, ratio_train)

    if not ventanas:
        return {"error": "No se pudieron generar ventanas walk-forward"}

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  WALK-FORWARD BACKTEST — Fiel a analyzer.py")
        print(f"  RS pesos: 1Y=20% 6M=20% 3M=25% 1M=25% 1W=10%")
        print(f"{'═'*70}")
        print(f"  Período:        {desde} → {hasta}")
        print(f"  Ventanas:       {len(ventanas)}")
        print(f"  Grid params:    {len(grid)} combinaciones")
        print(f"  Workers (CPU):  {n_cpu}")
        print(f"  Estado filtro:  {estado_filtro or 'todos'}")
        print(f"{'═'*70}")
        for v in ventanas:
            print(f"  Ventana {v['ventana']}: "
                  f"Train {v['train_desde']}→{v['train_hasta']}  "
                  f"Test {v['test_desde']}→{v['test_hasta']}")
        print()

    t0_total = time.time()
    resultados_ventanas = []

    for v in ventanas:
        vid = v["ventana"]

        # ── Pre-computar candidatos (1 escaneo por período) ───────────────────
        if verbose:
            print(f"  [V{vid}] Pre-computando train...", end=" ", flush=True)
        t0 = time.time()
        cands_train = precomputar_candidatos(
            historico, v["train_desde"], v["train_hasta"], n_workers=n_cpu)
        if verbose:
            print(f"{len(cands_train)} candidatos  {time.time()-t0:.1f}s")

        if verbose:
            print(f"  [V{vid}] Pre-computando test... ", end=" ", flush=True)
        t0 = time.time()
        cands_test = precomputar_candidatos(
            historico, v["test_desde"], v["test_hasta"], n_workers=n_cpu)
        if verbose:
            print(f"{len(cands_test)} candidatos  {time.time()-t0:.1f}s")

        # ── TRAIN: evaluar todos los params en paralelo (procesos) ────────────
        # run_backtest_rapido es CPU-bound puro Python → ThreadPoolExecutor
        # quedaba GIL-bound (1 core real, resto idle). ProcessPool con
        # candidatos+histórico vía initializer (1 vez por worker, amortizado
        # sobre los N params) da paralelismo real. Resultados IDÉNTICOS —
        # run_backtest_rapido es determinista y sin side-effects.
        #
        # Robustez (Windows): bajo presión acumulada de procesos (muchas
        # ventanas → muchos Pools spawneados) DuplicateHandle falla intermitente
        # con WinError 5. multiprocessing.Pool COLGABA esperando al worker
        # muerto. Fix: ProcessPoolExecutor (lanza BrokenProcessPool, no cuelga)
        # + retry 2x + fallback serial. La corrida nunca cuelga ni crashea;
        # peor caso una ventana corre serial (mismo _eval_train_params →
        # resultados idénticos, solo más lenta).
        if verbose:
            print(f"  [V{vid}] TRAIN {len(grid)} params...", end=" ", flush=True)
        t0 = time.time()

        def _train_serial():
            return [_eval_train_params(p, cands_train, historico, estado_filtro)
                    for p in grid]

        if n_cpu <= 1 or len(grid) < 2:
            train_results = _train_serial()
        else:
            import concurrent.futures as _cf
            hkeys, hdata = serializar_historico(historico)
            ctx = multiprocessing.get_context("spawn")
            # timeout escalado al grid: con 96 params una ventana puede tardar
            # >25min legítimamente (V2 tardó 1595s). 1800s fijo daba
            # TimeoutError FALSO (pool sano, solo lento) → retries inútiles +
            # fallback serial → timeout global. 120s/param ⇒ solo dispara en
            # cuelgue real. BrokenProcessPool detecta worker muerto al instante
            # igual, sin esperar este timeout.
            _map_timeout = max(3600, 120 * len(grid))
            train_results = None
            for intento in (1, 2, 3):
                try:
                    with _cf.ProcessPoolExecutor(
                            max_workers=min(n_cpu, len(grid)),
                            mp_context=ctx,
                            initializer=_init_train_worker,
                            initargs=(cands_train, hkeys, hdata, estado_filtro)) as ex:
                        train_results = list(ex.map(_eval_train_worker, grid,
                                                    timeout=_map_timeout))
                    break
                except Exception as e:
                    if verbose:
                        print(f"[pool intento {intento} falló: {type(e).__name__}]",
                              end=" ", flush=True)
                    # TimeoutError = pool sano pero lento → retry no ayuda,
                    # ir directo a serial. BrokenProcessPool/spawn = transitorio
                    # → sí reintentar.
                    if isinstance(e, _cf.TimeoutError):
                        break
            if train_results is None:
                if verbose:
                    print("[fallback serial]", end=" ", flush=True)
                train_results = _train_serial()

        elapsed_train = time.time() - t0
        if verbose:
            print(f"✓ {elapsed_train:.1f}s")

        validos = [r for r in train_results if r["n_ops"] >= 3]
        if not validos:
            if verbose:
                print(f"  Ventana {vid} — ⚠ Sin resultados válidos en train, usando defaults")
            best_train = {
                "params":        Params().as_dict(),
                "params_label":  Params().label(),
                "retorno_pct":   0, "max_dd_pct": 0,
                "profit_factor": 0, "n_ops":      0, "wr_pct": 0,
            }
        else:
            best_train = max(validos, key=_score_resultado)

        best_params = Params(**{k: v for k, v in best_train["params"].items()
                                if k in Params.__dataclass_fields__})

        # ── TEST: aplicar mejor params a datos nunca vistos ───────────────────
        r_test   = run_backtest_rapido(cands_test, historico, best_params, estado_filtro)
        pf_test  = r_test.get("portfolio", {})
        test_resultado = {
            "n_ops":            r_test.get("n_operaciones", 0),
            "retorno_pct":      pf_test.get("retorno_pct", 0) or 0,
            "max_dd_pct":       pf_test.get("max_drawdown_pct", 0) or 0,
            "max_dd_dur_d":     pf_test.get("max_dd_duration_d", 0) or 0,
            "win_rate_pct":     pf_test.get("win_rate_pct", 0) or 0,
            "profit_factor":    pf_test.get("profit_factor", 0) or 0,
            # A4 métricas avanzadas
            "sharpe":           pf_test.get("sharpe", 0) or 0,
            "sortino":          pf_test.get("sortino", 0) or 0,
            "calmar":           pf_test.get("calmar", 0) or 0,
            "expectancy_pct":   pf_test.get("expectancy_pct", 0) or 0,
            "max_win_streak":   pf_test.get("max_win_streak", 0) or 0,
            "max_loss_streak":  pf_test.get("max_loss_streak", 0) or 0,
        }

        if verbose:
            tr = best_train
            te = test_resultado
            print(f"  Ventana {vid} — Mejor train: {tr['params_label']}  "
                  f"ret={tr['retorno_pct']:+.1f}%  "
                  f"dd={tr['max_dd_pct']:.1f}%  "
                  f"pf={tr['profit_factor']:.2f}x  n={tr['n_ops']}")
            print(f"             Test:  ret={te['retorno_pct']:+.1f}%  "
                  f"wr={te['win_rate_pct']:.1f}%  "
                  f"pf={te['profit_factor']:.2f}x  n={te['n_ops']}\n")

        top5 = sorted(validos, key=_score_resultado, reverse=True)[:5]

        resultados_ventanas.append({
            "ventana":              vid,
            "train_desde":          str(v["train_desde"]),
            "train_hasta":          str(v["train_hasta"]),
            "test_desde":           str(v["test_desde"]),
            "test_hasta":           str(v["test_hasta"]),
            "params_optimos":       best_train["params_label"],
            "train_mejor":          best_train,
            "test_resultado":       test_resultado,
            "top5_train":           top5,
            "n_candidatos_train":   len(cands_train),
            "n_candidatos_test":    len(cands_test),
        })

    # ── Agregar resultados de test ────────────────────────────────────────────
    tests     = [v["test_resultado"] for v in resultados_ventanas]
    tests_ops = [t for t in tests if t["n_ops"] >= 1]

    def _med(k):
        vals = [t[k] for t in tests_ops if t.get(k) is not None]
        return round(statistics.mean(vals), 2) if vals else 0

    def _mdn(k):
        vals = sorted(t[k] for t in tests_ops if t.get(k) is not None)
        return round(vals[len(vals) // 2], 2) if vals else 0

    test_agregado = {
        "n_ventanas_test":       len(tests_ops),
        "ventanas_positivas":    sum(1 for t in tests_ops if t["retorno_pct"] > 0),
        "pct_ventanas_pos":      round(sum(1 for t in tests_ops if t["retorno_pct"] > 0) /
                                       max(len(tests_ops), 1) * 100, 1),
        "retorno_medio_pct":     _med("retorno_pct"),
        "retorno_mediana_pct":   _mdn("retorno_pct"),
        "max_dd_medio_pct":      _med("max_dd_pct"),
        "max_dd_dur_d_medio":    _med("max_dd_dur_d"),
        "win_rate_medio_pct":    _med("win_rate_pct"),
        "profit_factor_medio":   _med("profit_factor"),
        "profit_factor_mediana": _mdn("profit_factor"),
        # A4: métricas avanzadas (medias/medianas a través de ventanas test)
        "sharpe_medio":          _med("sharpe"),
        "sharpe_mediana":        _mdn("sharpe"),
        "sortino_medio":         _med("sortino"),
        "calmar_medio":          _med("calmar"),
        "calmar_mediana":        _mdn("calmar"),
        "expectancy_pct_medio":  _med("expectancy_pct"),
        "max_loss_streak_max":   max((t.get("max_loss_streak", 0) for t in tests_ops), default=0),
        "max_win_streak_max":    max((t.get("max_win_streak", 0) for t in tests_ops), default=0),
        "n_ops_total":           sum(t["n_ops"] for t in tests_ops),
        "n_ops_por_ventana":     round(sum(t["n_ops"] for t in tests_ops) /
                                       max(len(tests_ops), 1), 1),
    }

    # ── Análisis de robustez ──────────────────────────────────────────────────
    labels   = [v["params_optimos"] for v in resultados_ventanas]
    n_unicos = len(set(labels))
    estab    = 1 - (n_unicos - 1) / max(len(labels) - 1, 1)

    deltas_wr, deltas_pf = [], []
    for v in resultados_ventanas:
        tr = v["train_mejor"]
        te = v["test_resultado"]
        if te["n_ops"] >= 1:
            deltas_wr.append(te["win_rate_pct"] - tr.get("wr_pct", 0))
            deltas_pf.append(te["profit_factor"] - tr.get("profit_factor", 0))

    deg_wr = round(statistics.mean(deltas_wr), 2) if deltas_wr else 0
    deg_pf = round(statistics.mean(deltas_pf), 3) if deltas_pf else 0

    if deg_wr > -5 and deg_pf > -0.3 and estab > 0.4:
        veredicto = "✅ ROBUSTO — degradación aceptable entre train y test"
    elif deg_wr > -10 and deg_pf > -0.5:
        veredicto = "🟡 MODERADO — cierta degradación, revisar parámetros"
    else:
        veredicto = "🔴 OVERFITTING — degradación severa, sistema no generaliza"

    robustez = {
        "veredicto":             veredicto,
        "estabilidad_params":    round(estab, 3),
        "params_optimos_unicos": n_unicos,
        "labels_optimos":        labels,
        "degradacion_wr_media":  deg_wr,
        "degradacion_pf_media":  deg_pf,
    }

    elapsed = round(time.time() - t0_total, 1)

    return {
        "config": {
            "desde":         str(desde),
            "hasta":         str(hasta),
            "n_ventanas":    len(ventanas),
            "n_params_grid": len(grid),
            "n_workers":     n_cpu,
            "estado_filtro": estado_filtro,
            "elapsed_total": elapsed,
            "rs_pesos":      "1Y=20% 6M=20% 3M=25% 1M=25% 1W=10% (fiel a analyzer.py)",
        },
        "ventanas":      resultados_ventanas,
        "test_agregado": test_agregado,
        "robustez":      robustez,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORTE
# ═══════════════════════════════════════════════════════════════════════════════

def print_reporte_wf(resultado: dict):
    W   = 72
    cfg = resultado.get("config", {})
    agg = resultado.get("test_agregado", {})
    rob = resultado.get("robustez", {})

    print(f"\n{'═'*W}")
    print(f"  WALK-FORWARD — RESULTADOS")
    print(f"  {cfg.get('rs_pesos','')}")
    print(f"{'═'*W}")
    print(f"  Período:         {cfg.get('desde')} → {cfg.get('hasta')}")
    print(f"  Ventanas:        {cfg.get('n_ventanas')}")
    print(f"  Grid explorado:  {cfg.get('n_params_grid')} combinaciones")
    print(f"  Tiempo total:    {cfg.get('elapsed_total')}s")

    print(f"\n  {'RESULTADOS EN TEST (datos nunca vistos)':^{W-4}}")
    print(f"  {'─'*68}")

    vpos  = agg.get("ventanas_positivas", 0)
    vtot  = agg.get("n_ventanas_test", 0)
    ppct  = agg.get("pct_ventanas_pos", 0.0)
    rmed  = agg.get("retorno_medio_pct", 0.0)
    rmedn = agg.get("retorno_mediana_pct", 0.0)
    ddmed = agg.get("max_dd_medio_pct", 0.0)
    wrmed = agg.get("win_rate_medio_pct", 0.0)
    pfmed = agg.get("profit_factor_medio", 0.0)
    pfmdn = agg.get("profit_factor_mediana", 0.0)
    nops  = agg.get("n_ops_total", 0)
    nopv  = agg.get("n_ops_por_ventana", 0.0)

    print(f"  Ventanas con retorno positivo:  {vpos}/{vtot}  ({ppct:.1f}%)")
    print(f"  Retorno medio por ventana:      {rmed:+.1f}%  (mediana {rmedn:+.1f}%)")
    print(f"  Drawdown medio:                 {ddmed:.1f}%")
    print(f"  DD duración media:              {agg.get('max_dd_dur_d_medio',0):.0f}d")
    print(f"  Win rate medio:                 {wrmed:.1f}%")
    print(f"  Profit factor medio:            {pfmed:.2f}x  (mediana {pfmdn:.2f}x)")
    print(f"  Sharpe medio:                   {agg.get('sharpe_medio',0):.2f}  (mediana {agg.get('sharpe_mediana',0):.2f})")
    print(f"  Sortino medio:                  {agg.get('sortino_medio',0):.2f}")
    print(f"  Calmar medio:                   {agg.get('calmar_medio',0):.2f}")
    print(f"  Expectancy media:               {agg.get('expectancy_pct_medio',0):+.3f}%/op")
    print(f"  Max loss streak (peor):         {agg.get('max_loss_streak_max',0)} consec.")
    print(f"  Operaciones totales (test):     {nops}  ({nopv:.0f}/ventana)")

    print(f"\n  {'DETALLE POR VENTANA':^{W-4}}")
    print(f"  {'─'*68}")
    print(f"  {'V':>2}  {'Train':^22}  {'Test':^22}  {'Params óptimos'}")
    print(f"  {'─'*68}")

    for v in resultado.get("ventanas", []):
        tr    = v.get("train_mejor", {})
        te    = v.get("test_resultado", {})
        label = tr.get("params_label", "?")[:22]
        vid   = v.get("ventana")
        print(f"  {vid:>2}  "
              f"ret={tr.get('retorno_pct',0):+5.1f}% "
              f"dd={tr.get('max_dd_pct',0):4.1f}% "
              f"pf={tr.get('profit_factor',0):.2f}x  "
              f"ret={te.get('retorno_pct',0):+5.1f}% "
              f"dd={te.get('max_dd_pct',0):4.1f}% "
              f"pf={te.get('profit_factor',0):.2f}x  "
              f"{label}")

    print(f"\n  {'ANÁLISIS DE ROBUSTEZ':^{W-4}}")
    print(f"  {'─'*68}")
    print(f"  {rob.get('veredicto','')}")
    dwr = rob.get("degradacion_wr_media", 0.0)
    dpf = rob.get("degradacion_pf_media", 0.0)
    est = rob.get("estabilidad_params", 0.0)
    npu = rob.get("params_optimos_unicos", 0)
    print(f"  Degradación WR train→test:      {dwr:+.1f}pp")
    print(f"  Degradación PF train→test:      {dpf:+.2f}x")
    print(f"  Estabilidad parámetros:         {est:.2f}  "
          f"({npu} params distintos en {cfg.get('n_ventanas')} ventanas)")

    labels = rob.get("labels_optimos", [])
    if labels:
        from collections import Counter
        top_label, freq = Counter(labels).most_common(1)[0]
        print(f"\n  Parámetros más robustos (ganó en {freq}/{len(labels)} ventanas):")
        print(f"  → {top_label}")

    print(f"{'═'*W}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUARDAR RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════

def guardar_resultados_wf(resultado: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = REPORTS_DIR / f"walkforward_{ts}.json"

    exportable = {k: v for k, v in resultado.items()
                  if k not in ("train_all", "test_all")}

    ventanas_clean = []
    for v in exportable.get("ventanas", []):
        v_clean = {k: val for k, val in v.items() if k != "top5_train"}
        v_clean["top5_train"] = [
            {k2: v2 for k2, v2 in t.items() if k2 != "operaciones"}
            for t in v.get("top5_train", [])
        ]
        ventanas_clean.append(v_clean)
    exportable["ventanas"] = ventanas_clean

    with open(path, "w", encoding="utf-8") as f:
        json.dump(exportable, f, indent=2, default=str)
    print(f"  ✓ Walk-forward guardado en {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  DESCARGA DE HISTÓRICO
# ═══════════════════════════════════════════════════════════════════════════════

def descargar_historico(tickers, desde_extra_str, hasta_str):
    """
    Descarga con caché Parquet (bt_cache.py).
    La segunda ejecución lee del disco y es 10-20× más rápida.
    ETFs de sector se guardan con prefijo "ETF_".
    """
    if _CACHE_OK:
        return _cache_descargar(
            tickers     = list(tickers),
            desde_str   = desde_extra_str,
            hasta_str   = hasta_str,
            sector_etfs = SECTOR_ETFS,
            verbose     = True,
        )

    # Fallback sin caché
    import warnings, pandas as _pd
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance pandas"); return {}

    todos     = list(set(tickers + ["SPY"] + SECTOR_ETFS))
    historico = {}
    for i in range(0, len(todos), 50):
        batch = todos[i:i+50]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(batch, start=desde_extra_str, end=hasta_str,
                                  auto_adjust=True, progress=False,
                                  group_by="ticker", threads=True)
            if raw.empty: continue
            for t in batch:
                try:
                    df = raw[t].dropna(how="all") if len(batch) > 1 else raw.copy()
                    if isinstance(df.columns, _pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if not df.empty and len(df) > 20:
                        key = f"ETF_{t}" if t in SECTOR_ETFS else t
                        historico[key] = df
                except Exception: continue
        except Exception as e:
            print(f"  ⚠ Error batch: {e}")
        time.sleep(0.3)

    print(f"     ✓ {len(historico)} tickers descargados")
    return historico


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN / CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Walk-Forward Backtest — fiel a analyzer.py"
    )
    ap.add_argument("--desde",       default="2020-01-01",
                    help="Inicio del período (default: 2020-01-01)")
    ap.add_argument("--hasta",       default=str(datetime.date.today()),
                    help="Fin del período (default: hoy)")
    ap.add_argument("--ventanas",    type=int, default=5,
                    help="Número de ventanas walk-forward (default: 5)")
    ap.add_argument("--ratio-train", type=float, default=0.7,
                    help="Fracción train de cada ventana (default: 0.7)")
    ap.add_argument("--grid",        choices=["rapido","normal","completo"],
                    default="normal",
                    help="Tamaño del grid de parámetros (default: normal)")
    ap.add_argument("--estado",      default=None,
                    help="Filtrar por estado (BREAKOUT|SQUEEZE|EN BASE|...)")
    ap.add_argument("--workers",     type=int, default=None,
                    help="Número de workers CPU (default: auto)")
    ap.add_argument("--guardar",     action="store_true",
                    help="Guardar resultados en JSON")
    ap.add_argument("--tickers",     nargs="+", default=None,
                    help="Universo personalizado de tickers")
    ap.add_argument("--pit",         action="store_true",
                    help="Usar universo Point-in-Time (sin survivorship bias)")
    ap.add_argument("--rs-min",      type=int, default=None,
                    help="Forzar RS minimo en todo el grid (ej: 65)")
    ap.add_argument("--score-min",   type=int, default=None,
                    help="Forzar Score minimo en todo el grid (ej: 40)")
    ap.add_argument("--filtro-mercado", action="store_true", default=False,
                    help="Forzar filtro de mercado (SPY>SMA200) en todo el grid")
    ap.add_argument("--capital",       type=float, default=None,
                    help="Capital inicial USD (default: 10000)")
    args = ap.parse_args()

    desde = datetime.date.fromisoformat(args.desde)
    hasta = datetime.date.fromisoformat(args.hasta)

    # Compilar Numba una vez (evita pagar el coste en el primer chunk)
    if _CACHE_OK and tiene_numba():
        warmup_numba()
    elif not _CACHE_OK:
        print("[WF] ℹ️  bt_cache.py no encontrado — sin caché ni Numba.")

    # ── Descargar histórico ───────────────────────────────────────────────────
    if args.pit:
        print(f"\n[WF] Modo POINT-IN-TIME (sin survivorship bias)")
        try:
            from gpu_bt.pit_universe import PitUniverse
        except ImportError:
            print("ERROR: pit_universe.py no encontrado.")
            return
        pit = PitUniverse()
        pit.build(str(desde), str(hasta))
        historico = pit.historico
        if not historico:
            print("ERROR: PIT no pudo construir el histórico.")
            return
        print(f"[WF] Histórico PIT listo: {len(historico)} tickers")
    else:
        print(f"\n[WF] Modo UNIVERSO FIJO — Réplica fiel de analyzer.py")
        tickers     = args.tickers or UNIVERSO_FIJO
        desde_extra = str(desde - datetime.timedelta(days=420))  # margen para RS 1Y + SMA200
        print(f"[WF] Descargando histórico: {len(tickers)} tickers + {len(SECTOR_ETFS)} ETFs sector...")
        historico = descargar_historico(tickers, desde_extra, str(hasta))
        if not historico:
            print("ERROR: No se pudo descargar el histórico.")
            return
        print(f"[WF] Histórico listo: {len(historico)} tickers")

    # ── Grid de parámetros ────────────────────────────────────────────────────
    grid = build_param_grid(args.grid)

    # Filtrar grid si se forzaron minimos
    if args.rs_min is not None:
        antes = len(grid)
        grid = [p for p in grid if p.rs_minimo >= args.rs_min]
        print(f"[WF] --rs-min={args.rs_min}: {antes} -> {len(grid)} combinaciones")
    if args.score_min is not None:
        antes = len(grid)
        grid = [p for p in grid if p.score_minimo >= args.score_min]
        print(f"[WF] --score-min={args.score_min}: {antes} -> {len(grid)} combinaciones")
    if args.filtro_mercado:
        antes = len(grid)
        grid = [p for p in grid if p.filtro_mercado]
        print(f"[WF] --filtro-mercado: {antes} -> {len(grid)} combinaciones (solo SPY>SMA200)")
    if not grid:
        print("ERROR: El grid quedo vacio. Ajusta --rs-min / --score-min.")
        return

    if args.capital is not None:
        from dataclasses import replace as _dc_replace
        grid = [_dc_replace(p, capital_total=args.capital) for p in grid]
        print(f"[WF] --capital={args.capital:.0f}: aplicado a {len(grid)} combinaciones")

    print(f"[WF] Grid {args.grid}: {len(grid)} combinaciones de parametros")

    # ── Walk-Forward ──────────────────────────────────────────────────────────
    resultado = run_walk_forward(
        historico     = historico,
        desde         = desde,
        hasta         = hasta,
        n_ventanas    = args.ventanas,
        ratio_train   = args.ratio_train,
        param_grid    = grid,
        estado_filtro = args.estado,
        n_workers     = args.workers,
        verbose       = True,
    )

    print_reporte_wf(resultado)

    if args.guardar:
        guardar_resultados_wf(resultado)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # necesario en Windows
    main()