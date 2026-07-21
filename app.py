#!/usr/bin/env python3
"""
Phase 3 Paper Trading Engine — Web Deployment Version
=======================================================
Same trading logic as the original phase3_paper_trader.py, restructured so it
can run as an always-on web service on a free host (e.g. Render) with:
  - a protected /run endpoint that a free external cron pings once a day
  - a public "/" dashboard page showing current holdings + performance
  - state stored in Upstash Redis (REST API) instead of a local JSON file,
    because free web hosts wipe local disk on every restart/redeploy.

Environment variables required (set these in Render's dashboard, no yaml):
  UPSTASH_REDIS_REST_URL     - from your Upstash database page
  UPSTASH_REDIS_REST_TOKEN   - from your Upstash database page
  RUN_SECRET                 - any password you make up, protects /run
"""

import json
import os
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from curl_cffi import requests as curl_requests
from flask import Flask, request, abort, Response

# ----------------------------------------------------------------------
# CONFIG — identical to the original phase_3 script
# ----------------------------------------------------------------------
TICKERS = ["NVDA", "TQQQ", "SMH", "USD", "IBIT", "UPRO"]
BENCHMARK = "SPY"
STARTING_CAPITAL = 1000.0
TXN_COST = 0.0005
CASH_YIELD_ANNUAL = 0.045
DAILY_CASH_RATE = CASH_YIELD_ANNUAL / 252
HISTORY_DAYS = 500
TOP_N = 2

HORIZONS = [
    {"sma": 40,  "max": 12, "min": 6,  "weight": 0.35},
    {"sma": 120, "max": 22, "min": 8,  "weight": 0.65},
]

STATE_KEY = "phase3:paper_state"
RUN_SECRET = os.environ.get("RUN_SECRET", "change-me")

app = Flask(__name__)


# ----------------------------------------------------------------------
# 0. UPSTASH REDIS STATE STORE (replaces load_state/save_state file I/O)
# ----------------------------------------------------------------------
def _redis_url():
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        raise RuntimeError(
            "UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN are not set. "
            "Add them as environment variables in your host's dashboard."
        )
    return url, token


def redis_cmd(*parts):
    url, token = _redis_url()
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json=list(parts),
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("result")


def default_state():
    return {
        "created": datetime.now().strftime("%Y-%m-%d"),
        "starting_capital": STARTING_CAPITAL,
        "cash": STARTING_CAPITAL,
        "last_run": None,
        "positions": {},
        "closed_trades": [],
        "equity_curve": [],
        "spy_shares_ref": None,
        "last_dashboard": None,
    }


def load_state():
    raw = redis_cmd("GET", STATE_KEY)
    if not raw:
        return default_state()
    return json.loads(raw)


def save_state(state):
    redis_cmd("SET", STATE_KEY, json.dumps(state, default=str))


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def fetch_price_history():
    all_tickers = TICKERS + [BENCHMARK]
    session = curl_requests.Session(impersonate="chrome")
    raw = yf.download(all_tickers, period=f"{HISTORY_DAYS}d", session=session, progress=False)
    close_df = raw["Close"].ffill().bfill()
    return close_df


# ----------------------------------------------------------------------
# 2. SIGNAL ENGINE (unchanged from the original script)
# ----------------------------------------------------------------------
def compute_master_signals(close_df, tickers, horizons):
    close_vals = close_df[tickers].values
    shifted_close = close_df[tickers].shift(1).values
    num_days, num_assets = close_vals.shape
    master_signals = np.zeros_like(close_vals)

    for horizon in horizons:
        sma_w, t_max, t_min, strat_w = horizon["sma"], horizon["max"], horizon["min"], horizon["weight"]
        sma_vals = close_df[tickers].rolling(sma_w).mean().values
        trend_signals = np.zeros_like(close_vals)
        trend_states = np.zeros(num_assets)

        for t in range(max(sma_w, t_max), num_days):
            max_ch = np.max(shifted_close[t - t_max:t], axis=0)
            min_ch = np.min(shifted_close[t - t_min:t], axis=0)
            for asset in range(num_assets):
                if close_vals[t, asset] > max_ch[asset]:
                    trend_states[asset] = 1.0
                elif close_vals[t, asset] < min_ch[asset]:
                    trend_states[asset] = 0.0

                if close_vals[t, asset] <= sma_vals[t, asset]:
                    trend_signals[t, asset] = 0.0
                else:
                    trend_signals[t, asset] = trend_states[asset] * strat_w

        master_signals += trend_signals

    return master_signals


def compute_target_weights_today(close_df, tickers=TICKERS, horizons=HORIZONS, top_n=TOP_N):
    master_signals = compute_master_signals(close_df, tickers, horizons)
    num_assets = len(tickers)
    today_signal = master_signals[-1]

    rolling_vol = close_df[tickers].pct_change().rolling(21).std().fillna(0.01).values[-1]
    inv_vol = 1.0 / np.where(rolling_vol == 0, 0.01, rolling_vol)

    sma_120 = close_df[tickers].rolling(120).mean().values[-1]
    close_today = close_df[tickers].values[-1]
    trend_intensity = np.where(sma_120 > 0, (close_today - sma_120) / sma_120, 0)
    active_intensity = trend_intensity * (today_signal > 0).astype(float)

    top_indices = np.argsort(active_intensity)[-top_n:]
    rank_mask = np.zeros(num_assets)
    rank_mask[top_indices] = 1.0

    filtered_signals = today_signal * rank_mask
    weighted_signals = filtered_signals * inv_vol
    total_vol_weight = np.sum(weighted_signals)

    if total_vol_weight == 0:
        base_weights = np.zeros(num_assets)
    else:
        base_weights = weighted_signals / total_vol_weight

    total_active_signal = np.clip(np.sum(filtered_signals), 0.0, 1.0)
    final_weights = base_weights * total_active_signal

    return {t: float(w) for t, w in zip(tickers, final_weights)}


# ----------------------------------------------------------------------
# 3. REBALANCE (unchanged)
# ----------------------------------------------------------------------
def rebalance(state, target_weights, prices, run_date):
    state["cash"] += state["cash"] * DAILY_CASH_RATE

    equity = state["cash"] + sum(
        pos["shares"] * prices[t] for t, pos in state["positions"].items() if t in prices
    )

    for ticker in TICKERS:
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue

        target_value = equity * target_weights.get(ticker, 0.0)
        pos = state["positions"].get(ticker, {
            "shares": 0.0, "avg_cost": 0.0, "open_date": run_date,
            "realized_pl_accum": 0.0, "cost_basis_accum": 0.0,
        })
        current_value = pos["shares"] * price
        delta_value = target_value - current_value
        if abs(delta_value) < 1.0:
            continue

        delta_shares = delta_value / price
        friction_cost = abs(delta_value) * TXN_COST
        state["cash"] -= friction_cost

        if delta_shares > 0:
            new_shares = pos["shares"] + delta_shares
            pos["avg_cost"] = (
                (pos["shares"] * pos["avg_cost"] + delta_shares * price) / new_shares
                if new_shares > 0 else price
            )
            pos["cost_basis_accum"] += delta_shares * price
            if pos["shares"] == 0:
                pos["open_date"] = run_date
            pos["shares"] = new_shares
            state["cash"] -= delta_value
        else:
            sell_shares = min(-delta_shares, pos["shares"])
            realized = sell_shares * (price - pos["avg_cost"])
            pos["realized_pl_accum"] += realized
            pos["shares"] -= sell_shares
            state["cash"] += sell_shares * price

            if pos["shares"] <= 1e-6:
                total_cost_basis = pos["cost_basis_accum"] if pos["cost_basis_accum"] > 0 else 1e-9
                pl_pct = (pos["realized_pl_accum"] / total_cost_basis) * 100
                state["closed_trades"].append({
                    "ticker": ticker,
                    "open_date": pos["open_date"],
                    "close_date": run_date,
                    "pl": round(pos["realized_pl_accum"], 2),
                    "pl_pct": round(pl_pct, 2),
                    "win": pos["realized_pl_accum"] > 0,
                })
                pos = {"shares": 0.0, "avg_cost": 0.0, "open_date": None,
                       "realized_pl_accum": 0.0, "cost_basis_accum": 0.0}

        state["positions"][ticker] = pos

    state["positions"] = {t: p for t, p in state["positions"].items() if p["shares"] > 1e-6}
    state["last_run"] = run_date
    return state


# ----------------------------------------------------------------------
# 4. METRICS (unchanged)
# ----------------------------------------------------------------------
def compute_metrics(state, prices, spy_price):
    positions_out = []
    open_value = 0.0
    for ticker, pos in state["positions"].items():
        price = prices.get(ticker, pos["avg_cost"])
        mkt_val = pos["shares"] * price
        open_value += mkt_val
        unreal_pl = mkt_val - pos["shares"] * pos["avg_cost"]
        unreal_pl_pct = (unreal_pl / (pos["shares"] * pos["avg_cost"]) * 100) if pos["avg_cost"] > 0 else 0.0
        positions_out.append({
            "ticker": ticker,
            "shares": round(pos["shares"], 4),
            "avg_cost": round(pos["avg_cost"], 2),
            "last_price": round(price, 2),
            "market_value": round(mkt_val, 2),
            "unrealized_pl": round(unreal_pl, 2),
            "unrealized_pl_pct": round(unreal_pl_pct, 2),
        })

    equity = state["cash"] + open_value
    for p in positions_out:
        p["weight_pct"] = round((p["market_value"] / equity * 100) if equity else 0.0, 2)

    total_pl = equity - state["starting_capital"]
    total_pl_pct = (total_pl / state["starting_capital"]) * 100

    closed = state["closed_trades"]
    wins = [t for t in closed if t["win"]]
    losses = [t for t in closed if not t["win"]]
    win_rate = (len(wins) / len(closed) * 100) if closed else 0.0
    best = max((t["pl_pct"] for t in closed), default=0.0)
    worst = min((t["pl_pct"] for t in closed), default=0.0)

    if state.get("spy_shares_ref") is None and spy_price:
        state["spy_shares_ref"] = state["starting_capital"] / spy_price
    spy_equity = (state["spy_shares_ref"] * spy_price) if state.get("spy_shares_ref") and spy_price else None

    metrics = {
        "total_equity": round(equity, 2),
        "cash": round(state["cash"], 2),
        "total_pl": round(total_pl, 2),
        "total_pl_pct": round(total_pl_pct, 2),
        "win_rate_pct": round(win_rate, 2),
        "num_closed_trades": len(closed),
        "num_wins": len(wins),
        "num_losses": len(losses),
        "best_trade_pct": round(best, 2),
        "worst_trade_pct": round(worst, 2),
        "spy_equity": round(spy_equity, 2) if spy_equity else None,
        "spy_total_pl_pct": round((spy_equity - state["starting_capital"]) / state["starting_capital"] * 100, 2) if spy_equity else None,
    }
    return positions_out, metrics, equity, spy_equity


# ----------------------------------------------------------------------
# 5. THE DAILY JOB
# ----------------------------------------------------------------------
def run_job():
    run_date = date.today().strftime("%Y-%m-%d")
    state = load_state()

    if state.get("last_run") == run_date:
        return {"status": "already ran today", "run_date": run_date}

    close_df = fetch_price_history()
    prices = {t: float(close_df[t].dropna().iloc[-1]) for t in TICKERS if t in close_df}
    spy_price = float(close_df[BENCHMARK].dropna().iloc[-1]) if BENCHMARK in close_df else None

    target_weights = compute_target_weights_today(close_df)
    state = rebalance(state, target_weights, prices, run_date)

    # Extra read-only data for the dashboard's signal board. This does not
    # feed back into target_weights or rebalance() above — it just re-derives
    # each horizon's on/off state and momentum for display.
    fast_h, slow_h = HORIZONS[0], HORIZONS[1]
    fast_state = compute_master_signals(close_df, TICKERS, [fast_h])[-1]
    slow_state = compute_master_signals(close_df, TICKERS, [slow_h])[-1]
    sma_120 = close_df[TICKERS].rolling(120).mean().values[-1]
    close_today = close_df[TICKERS].values[-1]
    trend_intensity = np.where(sma_120 > 0, (close_today - sma_120) / sma_120, 0)
    signal_board = [
        {
            "ticker": t,
            "fast_on": bool(fast_state[i] > 0),
            "slow_on": bool(slow_state[i] > 0),
            "intensity_pct": round(float(trend_intensity[i]) * 100, 2),
            "weight_pct": round(target_weights[t] * 100, 1),
            "allocated": target_weights[t] > 0,
        }
        for i, t in enumerate(TICKERS)
    ]

    positions_out, metrics, equity, spy_equity = compute_metrics(state, prices, spy_price)
    state["equity_curve"].append({
        "date": run_date,
        "equity": round(equity, 2),
        "spy_equity": round(spy_equity, 2) if spy_equity else None,
    })
    state["last_dashboard"] = {
        "as_of": run_date,
        "positions": positions_out,
        "metrics": metrics,
        "target_weights_today": {k: round(v, 3) for k, v in target_weights.items()},
        "signal_board": signal_board,
    }

    save_state(state)
    return {"status": "ran", "run_date": run_date, "metrics": metrics}


# ----------------------------------------------------------------------
# 6. DASHBOARD PRESENTATION
# ----------------------------------------------------------------------
# Kept as a plain (non f-string) constant so every CSS brace is literal —
# no escaping needed. Only small, brace-free f-strings are used later to
# drop in actual numbers and rows.
DASHBOARD_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=JetBrains+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap');

:root {
  --bg: #0b0e14;
  --surface: #10141c;
  --surface-raised: #161b26;
  --border: #1e2530;
  --text: #e9ecf2;
  --text-dim: #8890a0;
  --text-faint: #4e5563;
  --fast: #f0b429;
  --slow: #4ea8de;
  --gain: #5fd4a8;
  --loss: #e8735c;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: 'IBM Plex Sans', sans-serif;
  -webkit-font-smoothing: antialiased;
}

.wrap { max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }

.mono { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
.gain { color: var(--gain); }
.loss { color: var(--loss); }

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  border-bottom: 1px solid var(--border);
  padding-bottom: 20px;
  margin-bottom: 28px;
}
.eyebrow {
  display: block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 6px;
}
.topbar h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  font-size: 26px;
  margin: 0;
}
.asof {
  display: flex;
  align-items: center;
  gap: 7px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--text-dim);
}
.dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--gain);
  box-shadow: 0 0 0 3px rgba(95, 212, 168, 0.15);
}

.hero {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.3fr);
  gap: 24px;
  margin-bottom: 20px;
}
@media (max-width: 760px) {
  .hero { grid-template-columns: 1fr; }
}

.hero-figure {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px 26px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.hero-label {
  font-size: 12px;
  color: var(--text-dim);
  margin-bottom: 10px;
}
.hero-number {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  font-size: 44px;
  line-height: 1.05;
  letter-spacing: -0.01em;
}
.hero-delta {
  margin-top: 10px;
  font-size: 13px;
  font-family: 'JetBrains Mono', monospace;
}
.hero-delta span { color: var(--text-dim); font-family: 'IBM Plex Sans', sans-serif; }

.signal-board {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
}
.board-label {
  display: block;
  font-size: 12px;
  color: var(--text-dim);
  margin-bottom: 12px;
}
.board-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
@media (max-width: 480px) {
  .board-grid { grid-template-columns: repeat(2, 1fr); }
}
.sig-cell {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--surface-raised);
}
.sig-cell--live {
  border-color: rgba(95, 212, 168, 0.35);
  background: linear-gradient(180deg, rgba(95, 212, 168, 0.07), rgba(95, 212, 168, 0.02));
}
.sig-ticker {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  font-size: 13px;
}
.sig-momentum {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  margin-top: 2px;
}
.sig-tags { display: flex; gap: 5px; margin-top: 8px; }
.tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 5px;
  border: 1px solid var(--border);
  color: var(--text-faint);
}
.tag--fast.on { color: #1a1305; background: var(--fast); border-color: var(--fast); }
.tag--slow.on { color: #071b26; background: var(--slow); border-color: var(--slow); }
.sig-weight {
  margin-top: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-dim);
}
.sig-weight--live { color: var(--gain); }

.stat-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
  margin-bottom: 24px;
}
.stat-chip {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
}
.stat-chip span {
  display: block;
  font-size: 11px;
  color: var(--text-dim);
  margin-bottom: 6px;
}
.stat-chip b {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  font-size: 17px;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 22px;
  margin-bottom: 20px;
}
.section-head {
  font-size: 12px;
  letter-spacing: 0.04em;
  color: var(--text-dim);
  margin-bottom: 14px;
  text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace;
}
.chart-shell { position: relative; height: 260px; }
.chart-legend {
  display: flex;
  gap: 18px;
  margin-top: 14px;
  font-size: 12px;
  color: var(--text-dim);
}
.chart-legend span { display: flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }

.tables { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 760px) { .tables { grid-template-columns: 1fr; } }

table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th, td {
  text-align: left;
  padding: 8px 6px;
  border-bottom: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace;
}
th {
  color: var(--text-faint);
  font-weight: 500;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
td.ticker-cell { font-weight: 600; }
td.muted-cell {
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--text-dim);
  text-align: center;
}

.empty-state {
  max-width: 420px;
  margin: 80px auto;
  text-align: center;
  color: var(--text-dim);
}
.empty-state h2 {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text);
}

footer {
  margin-top: 32px;
  font-size: 11.5px;
  color: var(--text-faint);
  border-top: 1px solid var(--border);
  padding-top: 16px;
}
"""

# Reads window.__CHART_DATA__ (injected separately as a one-line f-string
# script tag, right before this runs) — keeps this block brace-safe too.
DASHBOARD_JS = """
var d = window.__CHART_DATA__;
var ctx = document.getElementById('equityChart');
var gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 260);
gradient.addColorStop(0, 'rgba(95, 212, 168, 0.25)');
gradient.addColorStop(1, 'rgba(95, 212, 168, 0)');

new Chart(ctx, {
  type: 'line',
  data: {
    labels: d.labels,
    datasets: [
      {
        label: 'Strategy',
        data: d.equity,
        borderColor: '#5fd4a8',
        backgroundColor: gradient,
        fill: true,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25
      },
      {
        label: 'SPY buy & hold',
        data: d.spy,
        borderColor: '#4e5563',
        borderDash: [4, 3],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        tension: 0.25
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#4e5563', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
      y: { grid: { color: '#1e2530' }, ticks: { color: '#4e5563' } }
    }
  }
});
"""


# ----------------------------------------------------------------------
# 7. ROUTES
# ----------------------------------------------------------------------
@app.route("/run")
def run_endpoint():
    if request.args.get("key") != RUN_SECRET:
        abort(403)
    try:
        result = run_job()
        return result
    except Exception as e:
        return {"status": "error", "detail": str(e)}, 500


@app.route("/reset")
def reset_endpoint():
    if request.args.get("key") != RUN_SECRET:
        abort(403)
    save_state(default_state())
    return {"status": "reset"}


def _page_shell(body_html, extra_scripts=""):
    """Wraps body content with the static head/CSS (safe, unescaped braces)."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Phase 3 Paper Trader</title>"
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js'></script>"
        "<style>" + DASHBOARD_CSS + "</style></head><body><div class='wrap'>"
        + body_html + "</div>" + extra_scripts + "</body></html>"
    )


@app.route("/")
def dashboard():
    state = load_state()
    dash = state.get("last_dashboard")

    if not dash:
        body = (
            "<div class='empty-state'>"
            "<h2>No trades yet</h2>"
            "<p>The signal engine hasn't run for the first time. "
            "Once the daily cron job fires, positions and performance will show up here.</p>"
            "</div>"
        )
        return Response(_page_shell(body), mimetype="text/html")

    metrics = dash["metrics"]
    positions = dash["positions"]
    board = dash.get("signal_board", [])
    curve = state.get("equity_curve", [])
    closed = state.get("closed_trades", [])

    # --- signal board cells ---
    board_cells = []
    for c in board:
        live_class = " sig-cell--live" if c["allocated"] else ""
        mom_class = "gain" if c["intensity_pct"] >= 0 else "loss"
        fast_class = "tag--fast on" if c["fast_on"] else "tag--fast"
        slow_class = "tag--slow on" if c["slow_on"] else "tag--slow"
        weight_html = (
            f"<div class='sig-weight sig-weight--live'>{c['weight_pct']:.0f}% alloc</div>"
            if c["allocated"] else "<div class='sig-weight'>not allocated</div>"
        )
        board_cells.append(
            f"<div class='sig-cell{live_class}'>"
            f"<div class='sig-ticker'>{c['ticker']}</div>"
            f"<div class='sig-momentum {mom_class}'>{c['intensity_pct']:+.1f}% vs 120d</div>"
            f"<div class='sig-tags'>"
            f"<span class='tag {fast_class}'>40D</span>"
            f"<span class='tag {slow_class}'>120D</span>"
            f"</div>"
            f"{weight_html}"
            f"</div>"
        )
    board_html = "".join(board_cells) or "<div class='muted-cell'>No signal data yet</div>"

    # --- stat chips ---
    pl_class = "gain" if metrics["total_pl"] >= 0 else "loss"
    chips = [
        f"<div class='stat-chip'><span>Cash</span><b class='mono'>${metrics['cash']:,.2f}</b></div>",
        f"<div class='stat-chip'><span>Win rate</span><b class='mono'>{metrics['win_rate_pct']:.1f}%</b></div>",
        f"<div class='stat-chip'><span>Closed trades</span><b class='mono'>{metrics['num_closed_trades']} <span style='color:var(--text-dim); font-size:12px;'>({metrics['num_wins']}W/{metrics['num_losses']}L)</span></b></div>",
        f"<div class='stat-chip'><span>Best trade</span><b class='mono gain'>{metrics['best_trade_pct']:+.2f}%</b></div>",
        f"<div class='stat-chip'><span>Worst trade</span><b class='mono loss'>{metrics['worst_trade_pct']:+.2f}%</b></div>",
    ]
    if metrics.get("spy_total_pl_pct") is not None:
        spy_class = "gain" if metrics["spy_total_pl_pct"] >= 0 else "loss"
        chips.append(
            f"<div class='stat-chip'><span>SPY buy &amp; hold</span><b class='mono {spy_class}'>{metrics['spy_total_pl_pct']:+.2f}%</b></div>"
        )
    stat_row_html = "".join(chips)

    # --- positions table ---
    if positions:
        pos_rows = "".join(
            f"<tr><td class='ticker-cell'>{p['ticker']}</td>"
            f"<td>{p['shares']}</td><td>${p['avg_cost']:,.2f}</td>"
            f"<td>${p['last_price']:,.2f}</td><td>${p['market_value']:,.2f}</td>"
            f"<td class='{'gain' if p['unrealized_pl'] >= 0 else 'loss'}'>{p['unrealized_pl_pct']:+.2f}%</td>"
            f"<td>{p['weight_pct']:.1f}%</td></tr>"
            for p in positions
        )
    else:
        pos_rows = "<tr><td class='muted-cell' colspan='7'>Fully in cash — no open positions</td></tr>"

    # --- closed trades table ---
    if closed:
        trade_rows = "".join(
            f"<tr><td class='ticker-cell'>{t['ticker']}</td><td>{t['open_date']}</td><td>{t['close_date']}</td>"
            f"<td class='{'gain' if t['win'] else 'loss'}'>{t['pl_pct']:+.2f}%</td></tr>"
            for t in reversed(closed[-20:])
        )
    else:
        trade_rows = "<tr><td class='muted-cell' colspan='4'>No closed trades yet</td></tr>"

    hero_delta_class = "gain" if metrics["total_pl_pct"] >= 0 else "loss"

    body = f"""
<header class="topbar">
  <div>
    <span class="eyebrow">Phase 3 &middot; Trend Ensemble &middot; {', '.join(TICKERS)}</span>
    <h1>Paper portfolio</h1>
  </div>
  <div class="asof"><span class="dot"></span>Synced {dash['as_of']}</div>
</header>

<div class="hero">
  <div class="hero-figure">
    <span class="hero-label">Total equity</span>
    <div class="hero-number mono">${metrics['total_equity']:,.2f}</div>
    <div class="hero-delta {hero_delta_class}">{metrics['total_pl_pct']:+.2f}% <span>all-time, started at ${metrics['total_equity'] - metrics['total_pl']:,.0f}</span></div>
  </div>
  <div class="signal-board">
    <span class="board-label">Signal board &middot; 40-day / 120-day trend state</span>
    <div class="board-grid">{board_html}</div>
  </div>
</div>

<div class="stat-row">{stat_row_html}</div>

<div class="card">
  <div class="section-head">Equity curve</div>
  <div class="chart-shell"><canvas id="equityChart"></canvas></div>
  <div class="chart-legend">
    <span><span class="swatch" style="background:#5fd4a8;"></span>Strategy</span>
    <span><span class="swatch" style="background:#4e5563; border-top:1px dashed #4e5563; height:0;"></span>SPY buy &amp; hold</span>
  </div>
</div>

<div class="tables">
  <section class="card">
    <div class="section-head">Open positions</div>
    <table>
      <tr><th>Ticker</th><th>Shares</th><th>Avg cost</th><th>Last</th><th>Value</th><th>Unrl.</th><th>Weight</th></tr>
      {pos_rows}
    </table>
  </section>
  <section class="card">
    <div class="section-head">Recent closed trades</div>
    <table>
      <tr><th>Ticker</th><th>Opened</th><th>Closed</th><th>P/L</th></tr>
      {trade_rows}
    </table>
  </section>
</div>

<footer>Paper trading only &mdash; simulated with no real funds. Not investment advice.</footer>
"""

    chart_data = json.dumps({
        "labels": [c["date"] for c in curve],
        "equity": [c["equity"] for c in curve],
        "spy": [c.get("spy_equity") for c in curve],
    })
    scripts = f"<script>window.__CHART_DATA__ = {chart_data};</script><script>{DASHBOARD_JS}</script>"

    return Response(_page_shell(body, scripts), mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
