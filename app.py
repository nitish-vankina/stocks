#!/usr/bin/env python3
"""
Phase 3 Paper Trading Engine — Web Deployment Version (v2)
=======================================================
Same trading logic as the original phase3_paper_trader.py, restructured so it
can run as an always-on web service on a free host (e.g. Render) with:
  - a protected /run endpoint that a free external cron pings once a day
  - a public "/" dashboard page showing current holdings + performance,
    now with best-effort LIVE intraday repricing and a 30s auto-refresh
  - state stored in Upstash Redis (REST API) instead of a local JSON file,
    because free web hosts wipe local disk on every restart/redeploy.
  - a Telegram push notification fired every time /run actually executes

Environment variables required (set these in Render's dashboard, no yaml):
  UPSTASH_REDIS_REST_URL     - from your Upstash database page
  UPSTASH_REDIS_REST_TOKEN   - from your Upstash database page
  RUN_SECRET                 - any password you make up, protects /run and /reset
  TELEGRAM_BOT_TOKEN         - from @BotFather (optional — alerts silently skip if unset)
  TELEGRAM_CHAT_ID           - the chat/user/channel id to push alerts to (optional)
"""

import copy
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
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Palette used for the live "active allocation" bar (cycled per ticker).
ALLOC_PALETTE = ["#00ff9d", "#4ea8de", "#f0b429", "#c792ea", "#ff6b9d", "#5fd4a8"]
ALLOC_CASH_COLOR = "#3a3f4b"

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
        "cash_yield_accum": 0.0,
        "peak_equity": STARTING_CAPITAL,
    }


def load_state():
    try:
        raw = redis_cmd("GET", STATE_KEY)
    except Exception:
        # Upstash unreachable / misconfigured — fail soft into a fresh,
        # in-memory-only state so the dashboard can still render something.
        return default_state()
    if not raw:
        return default_state()
    try:
        state = json.loads(raw)
    except Exception:
        return default_state()
    # Backfill any new fields for state written by older versions of this app.
    defaults = default_state()
    for key, val in defaults.items():
        state.setdefault(key, val)
    return state


def save_state(state):
    redis_cmd("SET", STATE_KEY, json.dumps(state, default=str))


# ----------------------------------------------------------------------
# 0b. TELEGRAM ALERTS
# ----------------------------------------------------------------------
def send_telegram_alert(message):
    """Best-effort push notification. Never raises — a failed/unconfigured
    Telegram integration must never break the daily /run job."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        return resp.ok
    except Exception:
        return False


def build_alert_message(run_date, state, metrics, orders, positions_out, prev_equity):
    equity = metrics["total_equity"]
    daily_pl = equity - prev_equity
    daily_pl_pct = (daily_pl / prev_equity * 100) if prev_equity else 0.0

    lines = [
        f"*Phase 3 Paper Trader — {run_date}*",
        "",
        f"*Total Equity:* ${equity:,.2f}",
        f"*Daily P/L:* {daily_pl:+,.2f} ({daily_pl_pct:+.2f}%)",
        f"*All-Time P/L:* {metrics['total_pl_pct']:+.2f}%",
        f"*Cash Sweep Yield (to date):* ${metrics.get('cash_yield_collected', 0.0):,.2f}",
        "",
    ]

    if orders:
        lines.append("*Executed Rebalances:*")
        for o in orders:
            lines.append(f"  \u2022 {o['action']} {o['ticker']} — {o['shares']:.4f} sh (${o['value']:,.2f})")
    else:
        lines.append("_No rebalancing trades executed today._")

    lines.append("")
    lines.append("*Current Holdings:*")
    if positions_out:
        for p in positions_out:
            lines.append(
                f"  \u2022 {p['ticker']}: {p['weight_pct']:.1f}% "
                f"(${p['market_value']:,.2f}, {p['unrealized_pl_pct']:+.2f}%)"
            )
    else:
        lines.append("  \u2022 Fully in cash")

    if metrics.get("spy_total_pl_pct") is not None:
        lines.append("")
        lines.append(
            f"*Benchmark (SPY):* {metrics['spy_total_pl_pct']:+.2f}% "
            f"vs *Strategy:* {metrics['total_pl_pct']:+.2f}% "
            f"({metrics.get('benchmark_delta_pct', 0.0):+.2f}pp)"
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def fetch_price_history():
    """End-of-day history used by the signal engine (unchanged)."""
    all_tickers = TICKERS + [BENCHMARK]
    session = curl_requests.Session(impersonate="chrome")
    raw = yf.download(all_tickers, period=f"{HISTORY_DAYS}d", session=session, progress=False)
    close_df = raw["Close"].ffill().bfill()
    return close_df


def fetch_live_quotes(tickers):
    """Best-effort intraday price fetch for the dashboard.

    Tries 1-minute intraday bars first (real-time-ish while the market is
    open). If that comes back empty for a ticker — market closed, weekend,
    holiday, or a transient rate limit — falls back to the most recent daily
    close for that ticker. Returns a (possibly partial) {ticker: price} dict
    and never raises, so a market-data hiccup can't take the dashboard down.
    """
    prices = {}
    try:
        session = curl_requests.Session(impersonate="chrome")
        raw = yf.download(
            tickers, period="1d", interval="1m", session=session,
            progress=False, threads=True,
        )
        if raw is not None and not raw.empty and "Close" in raw:
            close_df = raw["Close"]
            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame(name=tickers[0])
            for t in tickers:
                if t in close_df and not close_df[t].dropna().empty:
                    prices[t] = float(close_df[t].dropna().iloc[-1])
    except Exception:
        pass

    missing = [t for t in tickers if t not in prices]
    if missing:
        try:
            session = curl_requests.Session(impersonate="chrome")
            raw2 = yf.download(
                missing, period="5d", interval="1d", session=session,
                progress=False, threads=True,
            )
            if raw2 is not None and not raw2.empty and "Close" in raw2:
                close_df2 = raw2["Close"]
                if isinstance(close_df2, pd.Series):
                    close_df2 = close_df2.to_frame(name=missing[0])
                for t in missing:
                    if t in close_df2 and not close_df2[t].dropna().empty:
                        prices[t] = float(close_df2[t].dropna().iloc[-1])
        except Exception:
            pass

    return prices


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
# 3. REBALANCE (extended: tracks cash-sweep yield accrual + executed orders)
# ----------------------------------------------------------------------
def rebalance(state, target_weights, prices, run_date):
    yield_amt = state["cash"] * DAILY_CASH_RATE
    state["cash"] += yield_amt
    state["cash_yield_accum"] = state.get("cash_yield_accum", 0.0) + yield_amt

    equity = state["cash"] + sum(
        pos["shares"] * prices[t] for t, pos in state["positions"].items() if t in prices
    )

    orders = []

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
            orders.append({"ticker": ticker, "action": "BUY", "shares": delta_shares, "value": delta_value})
        else:
            sell_shares = min(-delta_shares, pos["shares"])
            realized = sell_shares * (price - pos["avg_cost"])
            pos["realized_pl_accum"] += realized
            pos["shares"] -= sell_shares
            state["cash"] += sell_shares * price
            orders.append({"ticker": ticker, "action": "SELL", "shares": sell_shares, "value": sell_shares * price})

            if pos["shares"] <= 1e-6:
                total_cost_basis = pos["cost_basis_accum"] if pos["cost_basis_accum"] > 0 else 1e-9
                pl_pct = (pos["realized_pl_accum"] / total_cost_basis) * 100
                state["closed_trades"].append({
                    "ticker": ticker,
                    "open_date": pos["open_date"],
                    "close_date": run_date,
                    "close_price": round(price, 2),
                    "pl": round(pos["realized_pl_accum"], 2),
                    "pl_pct": round(pl_pct, 2),
                    "win": pos["realized_pl_accum"] > 0,
                })
                pos = {"shares": 0.0, "avg_cost": 0.0, "open_date": None,
                       "realized_pl_accum": 0.0, "cost_basis_accum": 0.0}

        state["positions"][ticker] = pos

    state["positions"] = {t: p for t, p in state["positions"].items() if p["shares"] > 1e-6}
    state["last_run"] = run_date
    return state, orders


# ----------------------------------------------------------------------
# 4. METRICS (extended: peak equity / drawdown, cash yield, benchmark delta)
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

    if len(losses) > 0:
        win_loss_ratio = len(wins) / len(losses)
    elif len(wins) > 0:
        win_loss_ratio = float("inf")
    else:
        win_loss_ratio = 0.0

    if state.get("spy_shares_ref") is None and spy_price:
        state["spy_shares_ref"] = state["starting_capital"] / spy_price
    spy_equity = (state["spy_shares_ref"] * spy_price) if state.get("spy_shares_ref") and spy_price else None
    spy_total_pl_pct = (
        round((spy_equity - state["starting_capital"]) / state["starting_capital"] * 100, 2)
        if spy_equity else None
    )

    # Peak equity / drawdown tracking.
    peak_equity = max(state.get("peak_equity", state["starting_capital"]), equity)
    state["peak_equity"] = peak_equity
    max_drawdown_pct = ((equity - peak_equity) / peak_equity * 100) if peak_equity else 0.0

    metrics = {
        "total_equity": round(equity, 2),
        "cash": round(state["cash"], 2),
        "total_pl": round(total_pl, 2),
        "total_pl_pct": round(total_pl_pct, 2),
        "win_rate_pct": round(win_rate, 2),
        "win_loss_ratio": win_loss_ratio,
        "num_closed_trades": len(closed),
        "num_wins": len(wins),
        "num_losses": len(losses),
        "best_trade_pct": round(best, 2),
        "worst_trade_pct": round(worst, 2),
        "spy_equity": round(spy_equity, 2) if spy_equity else None,
        "spy_total_pl_pct": spy_total_pl_pct,
        "benchmark_delta_pct": (
            round(total_pl_pct - spy_total_pl_pct, 2) if spy_total_pl_pct is not None else None
        ),
        "peak_equity": round(peak_equity, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "cash_yield_collected": round(state.get("cash_yield_accum", 0.0), 2),
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

    prev_equity = state["equity_curve"][-1]["equity"] if state.get("equity_curve") else state["starting_capital"]

    close_df = fetch_price_history()

    # Safely extract latest price for each ticker
    prices = {}
    for t in TICKERS:
        if t in close_df and not close_df[t].dropna().empty:
            prices[t] = float(close_df[t].dropna().iloc[-1])

    # Safely extract SPY price
    spy_price = None
    if BENCHMARK in close_df and not close_df[BENCHMARK].dropna().empty:
        spy_price = float(close_df[BENCHMARK].dropna().iloc[-1])

    target_weights = compute_target_weights_today(close_df)
    state, orders = rebalance(state, target_weights, prices, run_date)

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
        "orders": orders,
    }

    save_state(state)

    try:
        alert_msg = build_alert_message(run_date, state, metrics, orders, positions_out, prev_equity)
        send_telegram_alert(alert_msg)
    except Exception:
        pass  # a notification failure must never fail the job

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
  --bg: #111115;
  --surface: #1E1E24;
  --surface-raised: #26262e;
  --border: #2c2c36;
  --text: #e9ecf2;
  --text-dim: #8890a0;
  --text-faint: #565c6b;
  --fast: #f0b429;
  --slow: #4ea8de;
  --gain: #00ff9d;
  --loss: #ff4a4a;
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
  flex-wrap: wrap;
  gap: 12px;
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
  box-shadow: 0 0 0 3px rgba(0, 255, 157, 0.15);
}
.dot--stale {
  background: var(--text-faint);
  box-shadow: 0 0 0 3px rgba(86, 92, 107, 0.15);
}
.refresh-note {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-faint);
  margin-top: 2px;
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
  border-color: rgba(0, 255, 157, 0.35);
  background: linear-gradient(180deg, rgba(0, 255, 157, 0.08), rgba(0, 255, 157, 0.02));
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
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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

.alloc-bar {
  display: flex;
  width: 100%;
  height: 28px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.alloc-seg {
  height: 100%;
  min-width: 2px;
  transition: width 0.4s ease;
}
.alloc-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-top: 14px;
  font-size: 12px;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
}
.alloc-legend span { display: flex; align-items: center; gap: 6px; }
.alloc-swatch { width: 9px; height: 9px; border-radius: 3px; display: inline-block; }

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
gradient.addColorStop(0, 'rgba(0, 255, 157, 0.25)');
gradient.addColorStop(1, 'rgba(0, 255, 157, 0)');

new Chart(ctx, {
  type: 'line',
  data: {
    labels: d.labels,
    datasets: [
      {
        label: 'Strategy',
        data: d.equity,
        borderColor: '#00ff9d',
        backgroundColor: gradient,
        fill: true,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25
      },
      {
        label: 'SPY buy & hold',
        data: d.spy,
        borderColor: '#565c6b',
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
      x: { grid: { display: false }, ticks: { color: '#565c6b', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
      y: { grid: { color: '#2c2c36' }, ticks: { color: '#565c6b' } }
    }
  }
});
"""

# Small, dependency-free countdown so the page visibly ticks down to its
# next 30s auto-refresh (the actual reload is done by the <meta refresh> tag).
REFRESH_TICKER_JS = """
(function() {
  var secs = 30;
  var el = document.getElementById('refreshCountdown');
  if (!el) return;
  setInterval(function () {
    secs = secs > 0 ? secs - 1 : 0;
    el.textContent = secs + 's';
  }, 1000);
})();
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


def _page_shell(body_html, extra_scripts="", auto_refresh=True):
    """Wraps body content with the static head/CSS (safe, unescaped braces)."""
    refresh_meta = "<meta http-equiv='refresh' content='30'>" if auto_refresh else ""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        + refresh_meta +
        "<title>Phase 3 Paper Trader</title>"
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js'></script>"
        "<style>" + DASHBOARD_CSS + "</style></head><body><div class='wrap'>"
        + body_html + "</div>" + extra_scripts + "</body></html>"
    )


def _build_allocation_bar(positions_out, cash, equity):
    """Returns (bar_html, legend_html) for the active-allocation visual bar."""
    if equity <= 0:
        return "<div class='alloc-seg' style='width:100%; background:%s;'></div>" % ALLOC_CASH_COLOR, ""

    segments = []
    legend = []
    for i, p in enumerate(positions_out):
        if p["weight_pct"] <= 0:
            continue
        color = ALLOC_PALETTE[i % len(ALLOC_PALETTE)]
        segments.append(f"<div class='alloc-seg' style='width:{p['weight_pct']:.2f}%; background:{color};'></div>")
        legend.append(
            f"<span><span class='alloc-swatch' style='background:{color};'></span>{p['ticker']} {p['weight_pct']:.1f}%</span>"
        )

    cash_pct = max(0.0, (cash / equity * 100)) if equity else 0.0
    segments.append(f"<div class='alloc-seg' style='width:{cash_pct:.2f}%; background:{ALLOC_CASH_COLOR};'></div>")
    legend.append(
        f"<span><span class='alloc-swatch' style='background:{ALLOC_CASH_COLOR};'></span>Cash sweep {cash_pct:.1f}%</span>"
    )

    return "".join(segments), "".join(legend)


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
        return Response(_page_shell(body, auto_refresh=True), mimetype="text/html")

    all_tickers = TICKERS + [BENCHMARK]

    # --- best-effort live intraday repricing --------------------------------
    # We work on a deep copy so a live-price blip never corrupts the real,
    # persisted state (only /run is allowed to mutate + save that).
    live_prices = {}
    try:
        live_prices = fetch_live_quotes(all_tickers)
    except Exception:
        live_prices = {}

    working_state = copy.deepcopy(state)
    is_live = any(t in live_prices for t in TICKERS)

    if is_live:
        # Use a live price where we have one, otherwise fall back to the
        # last known close for that ticker so nothing silently drops to $0.
        prices = {}
        last_known = {p["ticker"]: p["last_price"] for p in dash.get("positions", [])}
        for t in TICKERS:
            if t in live_prices:
                prices[t] = live_prices[t]
            elif t in last_known:
                prices[t] = last_known[t]

        spy_price = live_prices.get(BENCHMARK)
        if spy_price is None:
            ref = working_state.get("spy_shares_ref")
            prev_spy_equity = dash["metrics"].get("spy_equity")
            if ref and prev_spy_equity is not None:
                spy_price = prev_spy_equity / ref
    else:
        prices = {p["ticker"]: p["last_price"] for p in dash.get("positions", [])}
        spy_price = None
        ref = working_state.get("spy_shares_ref")
        prev_spy_equity = dash["metrics"].get("spy_equity")
        if ref and prev_spy_equity is not None:
            spy_price = prev_spy_equity / ref

    try:
        positions_out, metrics, equity, spy_equity = compute_metrics(working_state, prices, spy_price)
    except Exception:
        positions_out, metrics = dash["positions"], dash["metrics"]
        equity = metrics["total_equity"]
        is_live = False

    board = dash.get("signal_board", [])
    curve = state.get("equity_curve", [])
    closed = state.get("closed_trades", [])

    # --- signal board cells (targets from the last daily rebalance) --------
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

    # --- stat chips (now including drawdown, cash yield, benchmark delta) --
    pl_class = "gain" if metrics["total_pl"] >= 0 else "loss"
    dd_class = "loss" if metrics.get("max_drawdown_pct", 0.0) < 0 else "gain"
    wl_ratio = metrics.get("win_loss_ratio", 0.0)
    wl_display = "∞" if wl_ratio == float("inf") else f"{wl_ratio:.2f}"

    chips = [
        f"<div class='stat-chip'><span>Cash</span><b class='mono'>${metrics['cash']:,.2f}</b></div>",
        f"<div class='stat-chip'><span>Peak equity</span><b class='mono'>${metrics.get('peak_equity', metrics['total_equity']):,.2f}</b></div>",
        f"<div class='stat-chip'><span>Max drawdown</span><b class='mono {dd_class}'>{metrics.get('max_drawdown_pct', 0.0):.2f}%</b></div>",
        f"<div class='stat-chip'><span>Cash yield collected</span><b class='mono gain'>${metrics.get('cash_yield_collected', 0.0):,.2f}</b></div>",
        f"<div class='stat-chip'><span>Win rate</span><b class='mono'>{metrics['win_rate_pct']:.1f}%</b></div>",
        f"<div class='stat-chip'><span>Win/Loss ratio</span><b class='mono'>{wl_display}</b></div>",
        f"<div class='stat-chip'><span>Closed trades</span><b class='mono'>{metrics['num_closed_trades']} <span style='color:var(--text-dim); font-size:12px;'>({metrics['num_wins']}W/{metrics['num_losses']}L)</span></b></div>",
        f"<div class='stat-chip'><span>Best trade</span><b class='mono gain'>{metrics['best_trade_pct']:+.2f}%</b></div>",
        f"<div class='stat-chip'><span>Worst trade</span><b class='mono loss'>{metrics['worst_trade_pct']:+.2f}%</b></div>",
    ]
    if metrics.get("spy_total_pl_pct") is not None:
        spy_class = "gain" if metrics["spy_total_pl_pct"] >= 0 else "loss"
        chips.append(
            f"<div class='stat-chip'><span>SPY buy &amp; hold</span><b class='mono {spy_class}'>{metrics['spy_total_pl_pct']:+.2f}%</b></div>"
        )
    if metrics.get("benchmark_delta_pct") is not None:
        delta_class = "gain" if metrics["benchmark_delta_pct"] >= 0 else "loss"
        chips.append(
            f"<div class='stat-chip'><span>Vs. benchmark</span><b class='mono {delta_class}'>{metrics['benchmark_delta_pct']:+.2f}pp</b></div>"
        )
    stat_row_html = "".join(chips)

    # --- active allocation bar -----------------------------------------------
    alloc_bar_html, alloc_legend_html = _build_allocation_bar(positions_out, metrics["cash"], equity)

    # --- positions table ------------------------------------------------------
    if positions_out:
        pos_rows = "".join(
            f"<tr><td class='ticker-cell'>{p['ticker']}</td>"
            f"<td>{p['shares']}</td><td>${p['avg_cost']:,.2f}</td>"
            f"<td>${p['last_price']:,.2f}</td><td>${p['market_value']:,.2f}</td>"
            f"<td class='{'gain' if p['unrealized_pl'] >= 0 else 'loss'}'>{p['unrealized_pl_pct']:+.2f}%</td>"
            f"<td>{p['weight_pct']:.1f}%</td></tr>"
            for p in positions_out
        )
    else:
        pos_rows = "<tr><td class='muted-cell' colspan='7'>Fully in cash — no open positions</td></tr>"

    # --- closed trades table (now with exit price + $ P/L) -------------------
    if closed:
        trade_rows = "".join(
            f"<tr><td class='ticker-cell'>{t['ticker']}</td><td>{t['open_date']}</td><td>{t['close_date']}</td>"
            f"<td>${t.get('close_price', 0):,.2f}</td>"
            f"<td class='{'gain' if t['win'] else 'loss'}'>${t['pl']:,.2f}</td>"
            f"<td class='{'gain' if t['win'] else 'loss'}'>{t['pl_pct']:+.2f}%</td></tr>"
            for t in reversed(closed[-20:])
        )
    else:
        trade_rows = "<tr><td class='muted-cell' colspan='6'>No closed trades yet</td></tr>"

    hero_delta_class = "gain" if metrics["total_pl_pct"] >= 0 else "loss"
    status_dot_class = "dot" if is_live else "dot dot--stale"
    status_label = "Live intraday" if is_live else f"Last close &middot; {dash['as_of']}"

    body = f"""
<header class="topbar">
  <div>
    <span class="eyebrow">Phase 3 &middot; Trend Ensemble &middot; {', '.join(TICKERS)}</span>
    <h1>Paper portfolio</h1>
  </div>
  <div>
    <div class="asof"><span class="{status_dot_class}"></span>{status_label}</div>
    <div class="refresh-note">auto-refresh in <span id="refreshCountdown">30s</span></div>
  </div>
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
  <div class="section-head">Active allocation</div>
  <div class="alloc-bar">{alloc_bar_html}</div>
  <div class="alloc-legend">{alloc_legend_html}</div>
</div>

<div class="card">
  <div class="section-head">Equity curve</div>
  <div class="chart-shell"><canvas id="equityChart"></canvas></div>
  <div class="chart-legend">
    <span><span class="swatch" style="background:#00ff9d;"></span>Strategy</span>
    <span><span class="swatch" style="background:#565c6b; border-top:1px dashed #565c6b; height:0;"></span>SPY buy &amp; hold</span>
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
      <tr><th>Ticker</th><th>Opened</th><th>Closed</th><th>Exit</th><th>P/L</th><th>P/L %</th></tr>
      {trade_rows}
    </table>
  </section>
</div>

<footer>Paper trading only &mdash; simulated with no real funds. Not investment advice. Intraday prices are best-effort and may lag or briefly be unavailable outside market hours.</footer>
"""

    chart_data = json.dumps({
        "labels": [c["date"] for c in curve],
        "equity": [c["equity"] for c in curve],
        "spy": [c.get("spy_equity") for c in curve],
    })
    scripts = (
        f"<script>window.__CHART_DATA__ = {chart_data};</script>"
        f"<script>{DASHBOARD_JS}</script>"
        f"<script>{REFRESH_TICKER_JS}</script>"
    )

    return Response(_page_shell(body, scripts, auto_refresh=True), mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
