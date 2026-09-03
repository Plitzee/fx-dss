# Re-planning prompt for Claude Code — FX pattern-rule discovery system

> Paste this whole file as your first message, or save it in the repo and say:
> "Read docs/PROMPT_REPLAN_EN.md and do exactly what it asks."

---

## 0. What I want from you in THIS session

**Do not write any production code yet.** I want a written plan I can review and
approve first. Deliver a plan document (`docs/REPLAN_2026.md`) plus a short
summary in chat. Only after I say "approved" do you start building.

Before planning, **read the existing repo** — do not rely on my summary below,
verify it yourself:

```
README.md
docs/               (all .md files — especially DANHGIA_CUOI.md, KHOA_SO.md,
                     TANG2_BIENDONG.md, TANG6_HIEU_CHUAN.md, TANG6B_DUNGTOIUU.md,
                     SIZING_COMPARISON.md, TICHHOP_HUYH.md, TOANMACH_E2E.md,
                     TANG4_DANHMUC.md, DONGBO_SANXUAT.md, MAU_HINH_FX.md,
                     TAI_LIEU_LIEN_QUAN.md)
src/split.py        (the official train/valid/test protocol — obey it)
src/volfc2.py       (volatility forecasting, currently the strongest component)
src/huyh_patterns.py, src/run_symbolic.py, src/run_sax_stats.py
src/decision_record.py, src/position_sizing.py, src/optimal_stop.py
data/               (list what exists and what each file contains)
web/                (the small Next.js + Python-serverless UI already built)
```

Existing docs and code comments are in Vietnamese. Keep it that way for anything
you write into the repo; talk to me in Vietnamese in chat. This prompt is in
English only because it is a spec.

---

## 1. The project, restated

This is my MIS graduation thesis. The direction has changed and I want the whole
system re-planned around the new goal.

**New core goal — cross-dataset rule mining, then transfer:**

1. Take many datasets from **one domain** (right now: spot FX rates of several
   currencies against USD — 6 pairs, ~14 years of data already in the repo).
2. Train / data-mine **across all of them together** to extract a **list of
   general rules** — patterns that hold across pairs, not curve-fitted to one.
3. Take that rule list and apply it to (a) a **new, unseen dataset** (a pair
   never trained on) and (b) the **future** of the pairs already trained on.
4. Each rule has the form:
   *"When signal A is present → next-horizon outcome distribution is
   X% down / Y% flat / Z% up"*; a different pattern B gives a different
   distribution.

**The product output is a calibrated probability, not a buy/sell call.** The UI
shows exactly three numbers: P(down) in red, P(flat/low-movement) in yellow,
P(up) in green. Getting those three percentages honest and well-calibrated is
phase 1 and the only thing I care about right now.

**On the literature:** I already had the FX-pattern papers researched (see
`docs/MAU_HINH_FX.md`, `docs/TAI_LIEU_LIEN_QUAN.md`). I cannot download those
papers' datasets and re-run them. So we learn the *method* from them and
re-implement it on **our own data, with our own training runs**. Every number in
this project must come from a run in this repo. No number copied from a paper is
ever presented as our result.

---

## 2. Hard constraints — negative results already established in this repo

These were paid for with real experiments. **The plan must respect them, and must
not silently re-discover or contradict them without new evidence.** If your plan
implies one of them is wrong, say so explicitly and design the experiment that
would overturn it.

1. **Direction is not predictable at daily horizon on this data.** Layer 1 was
   built, tested and rejected: E[z_T] ≈ 0. Any plan that assumes an up/down edge
   must first re-establish it, with a pre-registered test.
2. **Volatility IS predictable** and is the strongest asset we have (HAR family /
   STHARQ ensemble, large QLIKE improvements). Whatever is genuinely forecastable
   here is mostly *magnitude*, not *sign*.
3. **Symbolic/SAX sequential pattern mining survived only as volatility
   patterns.** Out of 4,722 candidate length-3 patterns, after a 4-stage filter
   (common → leave-one-pair-out → out-of-time 2022–2026) exactly **3** survived,
   and all 3 predict volatility state. **Zero direction patterns survived.**
4. **Reinforcement learning failed** for the conditioning task (PPO amplitude
   0.018, CVaR-PPO 0.030 vs 0.800 for a two-line hand rule); a tabular contextual
   bandit did better (0.144) but still lost, and bought growth with tail risk.
   Domain knowledge beat learned policies. Do not propose deep RL as a core
   component without addressing this.
5. **Fuzzy logic added nothing** over a product of two linear factors (+0.08%,
   inside seed noise).
6. **Momentum entries inside the high-volatility regime lose money with
   statistical significance** (Sharpe −0.62, p=0.001, survives Bonferroni). This
   is a real, usable *negative* signal.
7. **Conformal prediction under-covers while the account is in drawdown**
   (~89.3% vs 90.3% nominal-at-peak) — measured, unfixed, must be disclosed.

**Data-integrity protocol that must be carried over:**

- Official split in `src/split.py`: train `< 2021-10-13`, validation
  `< 2023-11-20`, test after. **Model and hyper-parameter selection may only use
  the validation segment.** The test segment is scored once.
- `docs/KHOA_SO.md` defines a **locked hold-out** (6 additional cross pairs +
  all of 2026) that has never been opened. It stays closed until the final run,
  and every dev-set experiment must be logged before it is opened. This is our
  only genuinely untouched data and it is the right home for the
  "apply the rule list to a brand-new dataset" claim.
- Everything is causal / expanding-window. No look-ahead, ever. Any leak test
  that exists must keep running.

---

## 3. Data currently available (verify and extend)

In `data/`: `panel2_6pairs.csv` (Date, pair, sig, zT, zL, zH, rv5 — ~3,600
sessions × 6 pairs, 2012→2025), `prices/{PAIR}_d1.csv` and `_h1.csv` (OHLC from
2010), `rv_adv.csv`, `rv_multi.csv` (realised variance, multiple estimators),
`spread_hourly_all.csv`, `slippage.csv` (from 60,617 real stop touches),
`cost_table.csv`, `carry.csv` (monthly rate differentials), `cb_dates.csv`
(central-bank calendar, expires end-2026), `fred_rates.csv`, `dukas_volume.csv`.
Raw M1 bars come from a Dukascopy downloader in the parent folder.

The plan must state: what is enough, what is missing, and what must be added
(more pairs? more asset classes to test cross-domain transfer? a news corpus?).

---

## 4. What the plan must contain

Write `docs/REPLAN_2026.md` with these sections. Be concrete — file names,
methods, sample sizes, decision criteria — not generalities.

### 4.1 Problem formalisation
- Exact definition of the 3 classes. The "flat" band must be **volatility-scaled**
  (e.g. |r| < k·σ̂ where σ̂ comes from the existing layer-2 forecast), not a fixed
  pip threshold. Justify k and the forecast horizon (1 day? 5 days? intraday?),
  and plan a sensitivity analysis over both.
- Be explicit about this: because direction is unpredictable but magnitude is,
  most of the achievable skill will live in the **flat vs. not-flat** split, and
  the **up vs. down** split will sit near base rates. The plan must say how the
  system reports that honestly instead of manufacturing false confidence.
- State the prediction target formally, plus the loss (log-loss / Brier) and the
  baselines that must be beaten.

### 4.2 Rule-mining methodology (the heart of the thesis)
Compare at least 4 families on the same data, same protocol, same folds:
- symbolic / SAX sequential patterns (extend the existing HuyH branch),
- motif discovery (matrix profile) on normalised windows,
- interpretable rule learners (CART→rule list, RuleFit, skope-rules, or a
  Bayesian rule list) over engineered features,
- gradient boosting as a strong non-interpretable ceiling, with SHAP used to
  distil candidate rules — reported as an upper bound, not shipped as "the rule".
- optionally: regime models (HMM / change-point) as a rule-conditioning layer.

For every mined rule, the plan must require reporting: support n, class
distribution, lift over base rate, per-pair breakdown, out-of-time stability, and
a **multiple-testing correction** (FDR/Bonferroni; ideally White's Reality Check
or Hansen's SPA, since we will screen thousands of candidates). Data snooping is
the number-one risk in this project — the plan must have an explicit defence.

**Generality test (this is the thesis's actual claim):** leave-one-pair-out —
mine on 5 pairs, evaluate on the 6th, then finally on the locked cross pairs.
A rule that does not transfer is not a "general rule" and must be discarded or
labelled pair-specific.

### 4.3 Calibration
The delivered product is three percentages, so calibration *is* the product.
Plan for: reliability diagrams, ECE, Brier skill score vs climatology, per-regime
and per-pair calibration, and a calibration method (Platt / isotonic /
Venn-Abers / the conformal machinery already in `src/decision_record.py`).
Mandatory baselines: unconditional class frequencies (climatology), persistence,
and a σ̂-only model with no patterns. **If the mined rules do not beat the
σ̂-only model, the plan must require us to report that plainly.**

### 4.4 Backtest / evaluation protocol
Walk-forward with expanding windows, per-pair and pooled; the existing
Diebold-Mariano + Newey-West apparatus should be reused for significance;
one-shot scoring on the test segment; locked hold-out opened once at the end.
State exactly what would count as failure.

### 4.5 The news / event module
I want the system to react to things like "US imposes tariffs on country X",
central-bank decisions, geopolitical shocks. Design it **as an event study, not
as a magic predictor**:
- an event taxonomy (rate decision, CPI/NFP surprise, tariff/trade action,
  geopolitical escalation, intervention), with a labelled historical event list,
- measured historical response: distribution of returns and σ around each event
  type, per pair, with n and confidence intervals,
- at run time the system says *"events of this type have historically been
  followed by <measured distribution>; sample size n; this is history, not a
  forecast"*, and optionally widens the flat band / abstains around scheduled
  events,
- an LLM may be used to **classify and tag incoming news into the taxonomy**, but
  must never be the thing that outputs a probability.
- Say honestly what data we would need (a historical news corpus with timestamps)
  and whether we can actually obtain it; if we cannot, scope the module down to
  the scheduled-event calendar we already have (`cb_dates.csv`) and say so.

### 4.6 UI specification
Trading-platform-grade, browser-based:
- login / user accounts (state the auth + DB choice and why),
- candlestick charts per pair with multiple timeframes, pan/zoom
  (recommend a library — e.g. TradingView `lightweight-charts` — and check its
  licence),
- a set of technical indicators and overlays similar to what traders expect
  (EMA/SMA, RSI, MACD, Bollinger, ATR, volume, S/R levels, regime shading).
  **Note:** LuxAlgo is a proprietary paid TradingView indicator suite — we can
  build our *own* equivalents with open, documented formulas, clearly labelled as
  ours; we must not clone or claim theirs. Plan accordingly.
- the three-box prediction panel underneath: red P(down) / yellow P(flat) /
  green P(up), each with the percentage, the sample size behind it, and which
  rules fired,
- an "explain" view: which patterns matched, their historical statistics, and the
  measured uncertainty of the number itself,
- an events/news panel driven by 4.5,
- optional advanced panel reusing the already-validated risk layers
  (position sizing, hold/close DP) — do not throw that work away, demote it.

Also decide: keep the current Next.js + Python-serverless skeleton in `web/`, or
restructure. Say why, and what the API contract becomes.

### 4.7 Architecture, repo layout, migration
Show the target module/folder layout, what is reused from the current `src/`
(layer 2 volatility, conformal calibration, slippage/cost models, split
protocol), what is retired, and what is new. Include a data-flow diagram
(Mermaid is fine). Explain how offline training artefacts reach the online API.

### 4.8 Plan of work
Phased milestones with a clear definition of done per phase, ordered so that the
riskiest scientific question (do transferable rules with real lift exist at all?)
is answered **before** the expensive UI work. Include an explicit
"kill criteria" section: what result would tell us the whole rule-mining premise
does not hold, and what we would ship instead in that case.

### 4.9 Thesis mapping
Map each phase to thesis chapters, and mark clearly which parts are a **research
contribution** (methodology, cross-dataset transfer results, negative findings)
versus **engineering** (UI, API, deployment).

### 4.10 Open questions for me
List the decisions you need from me, each with your recommended option and the
trade-off. Do not start building until I answer them.

---

## 5. How I want you to work

- **Honesty over optimism.** Negative results are results and get published in the
  thesis. Never dress up a weak finding. Never invent a number.
- **Every number must be reproducible** by a script in this repo, and must be
  explained in plain Vietnamese the moment it is presented — including what it
  means and why it matters — without me having to ask.
- Keep the existing statistical-rigour habits: expanding windows, DM tests with
  Newey-West, multiple-testing corrections, selection only on validation,
  one-shot test scoring, locked hold-out untouched.
- Ask me before doing anything irreversible or expensive.
- Write the plan first. I approve. Then we build.
