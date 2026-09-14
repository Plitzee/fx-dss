# UNIVERSITY OF INFORMATION TECHNOLOGY
## ADVANCED PROGRAM IN INFORMATION SYSTEMS

# THESIS PROPOSAL

**THESIS TITLE: BUILDING A STATISTICALLY VALIDATED DECISION SUPPORT SYSTEM FOR FOREIGN EXCHANGE TRADING WITH CALIBRATED UNCERTAINTY AND RISK CONTROL**

**Advisor:** ASSOCIATE PROFESSOR NGUYEN DINH THUAN

**Implementation period:** 09/2026 to 12/2026 (16 weeks)

**Students:**

NGUYEN QUOC HUY – 22520566

HUYNH TRAN QUOC HUY – 22520544

---

## Introduction

Retail FX decision-support tools typically display a directional forecast and a confidence label whose empirical foundation is weaker than their presentation suggests: daily returns have resisted prediction since Meese and Rogoff [1], and technical-rule replications show in-sample profitability largely disappearing out of sample, absorbed by a momentum factor [2], [3]. The core difficulty is inference under a low signal-to-noise ratio with a large hypothesis space — evaluating thousands of candidate rules finds spurious relationships unless the hypothesis space is enumerated in advance and the family-wise error rate is controlled [4]–[6] — compounded by the fact that a negative finding is uninformative without its minimum detectable effect size (MDES). Predictability also differs by axis: the HAR model of realized volatility [7]–[9] gives well-established *magnitude* forecasts while *direction* remains contested [10], [11], so a system should state explicitly which axis it claims skill on.

Preliminary work has established feasibility through three pre-registered phases, each closed with a written decision record. A complete pipeline covers six pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF) over 2010–2025 (≈34.9M one-minute bars); a production volatility layer, a three-class probability layer, and a risk layer run end to end through a web application updated daily. The extended HAR model attains test-set QLIKE 0.1585 vs. 0.2172 for the prior baseline, and 14 ML/DL alternatives plus 509 combinations fail to displace it from the Model Confidence Set [12]. On direction, 8,652 rule hypotheses across twelve discovery branches yield zero survivors after family-wise error control. A pre-registered causal-vs-validation feature-selection ablation is completed and frozen; its sealed-set evaluation was withheld because the pre-specified opening rule was not met.

This thesis is therefore as much an **engineering** problem as a statistical one: a decision support system whose every number is traceable to a pre-registered measurement, delivered as software that runs continuously rather than a script for a report. The data pipeline, computation layer, API, and web application (see *Proposed Method and System Architecture*) are a primary deliverable, developed across all sixteen weeks rather than assembled at the end.

## Market Survey

Existing retail tools were surveyed to state the contribution relative to what exists. **Signal marketplaces** (MQL5/MetaTrader, TradingView) publish track records from the vendor's own in-sample selection, with no multiple-testing correction. **Portfolio-analytics services** (Myfxbook, FX Blue) report realized statistics but no calibrated forward-looking probabilities. **Broker "confidence" widgets** attach a qualitative label with no scoring rule. None discloses the hypothesis space searched, the multiple-testing correction, a calibration guarantee, or a quantified probability of ruin — gaps identified in decision support systems generally [27] and specifically targeted here.

## Research Objectives

Design, implement, and validate a decision support system in which forecast skill is established per target axis under multiple-testing control, uncertainty is communicated via calibrated conformal prediction, and position sizing is constrained by an estimated probability of ruin. Three quantity groups are kept distinct: **predictive axes** (direction, magnitude, tail probability), **calibration diagnostics** (coverage, reliability), and the **decision layer** (VaR, ES, stop-out probability, sizing).

- **Axis-specific predictability.** Under one fixed protocol, determine on which axis out-of-sample skill exists, reported separately.
- **Model-complexity benchmarking.** Compare HAR-type, tree-based, recurrent, attention-based, and foundation-model alternatives under proper scoring rules and formal tests.
- **Incremental information value.** Test whether central-bank text, exogenous/cross-asset variables, and candlestick geometry [26] add value, and whether causal filtering beats ordinary selection.
- **Calibrated uncertainty.** Validate adaptive conformal prediction for target coverage under distribution shift.
- **Risk-controlled decisions.** Produce a per-session record: volatility, stop distance, stop-out probability, VaR/ES with backtests, ruin-constrained sizing.
- **Power-bounded negative findings.** Report the MDES for every negative result.
- **Engineering a production architecture.** Design and extend, across all sixteen weeks, a five-layer software architecture (pipeline, computation layer, versioned artefacts, API, web application) so every number stays traceable.

## Research Scope

Daily-horizon decision support for six major USD pairs. The 2010–2025 sample splits into training, validation (selection only), and a held-out development-test segment scored once. A **sealed evaluation set** — six cross pairs never used in development (EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY, EUR/CHF, NZD/USD) plus 2026 data for all twelve pairs — opens exactly once after the configuration is frozen, with no retuning after.

The primary source is one-minute bar data, from which daily bars and five-minute realized measures are computed; supplementary sources include the central-bank/macro calendar, exogenous daily series, and tick-derived transaction costs. Horizons are one, five, and twenty sessions (one-session default). The study explicitly does **not** target a profitable trading strategy — execution, microstructure, and sub-daily timing are out of scope; causality claims are limited to Granger-sense temporal precedence.

## Research Subjects

**The statistical validation of decision support outputs for FX under a low signal-to-noise ratio**: how skill is attributed to an axis and an information layer; how multiple-testing control, power analysis, and sealed-set validation combine to produce replicable claims; how calibrated forecasts convert into risk limits with a measured ruin probability; and how this evidence should be presented so a user can separate what is measured from what is assumed.

## Proposed Method and System Architecture

Each item below is labelled **(implemented)** or **(proposed)**; nothing is a finished result until it appears with measured numbers in Expected Results.

### Proposed Predictive and Risk Layers

- **Magnitude — extended HAR.** *(implemented)* Heterogeneous-autoregressive on realized variance [7], extended with measurement-error correction [8] and jump/semivariance decomposition [9], selected over 14 ML/DL challengers and 509 combinations under the Model Confidence Set [12].
- **Direction/probability — online expert combination.** *(implemented)* Hedge-style combination [13] of four sub-models, re-weighted every session; challenged by linear, tree, recurrent, attention, and foundation-model alternatives.
- **Causal discovery.** *(implemented — validation role)* Five triangulated methods decide which exogenous variable is trusted; using the effect to adjust the forecast directly was tested and found not yet effective (Deferred / Future Work).
- **Calibration — Mondrian-stratified ACI.** *(implemented)* [19], [20], 1.2-point maximum deviation vs. 2.4–3.2 for alternatives. *(proposed)* PID-controller online update [30] for below-target coverage during drawdown.
- **Risk/decision.** *(implemented)* Empirical-quantile VaR/ES with Kupiec [21], Christoffersen [22], dynamic-quantile [23] backtests and a consistent scoring function [24]; Kelly-derived sizing [25] capped by ruin probability. *(proposed)* Portfolio-correlation adjustment and a measured slippage coefficient.

### Causal Discovery Method

The exogenous layer needs a method conditioning on many candidates simultaneously, separating a direct link from a shared confound, under the same false-discovery control as the pattern families. The **official gate** is Granger-type F-testing under step-down Westfall–Young control [4]. Four independent methods triangulate the result — the answer to "which algorithm, and how do you know it is causality, not correlation":

| Method | Question | Key result |
|---|---|---|
| **PCMCI** [18], [29] | Survives simultaneous conditioning? | VIX survives on every pair; gold-implied volatility (GVZCLS) loses significance once conditioned on VIX — the confound-vs-cause distinction pairwise Granger cannot make |
| **Double ML** [33] | Effect *size*, nonlinear nuisance model? | Positive, Holm-significant in **6/6** pairs — strongest result |
| **Causal Forest** [34] | Regime-dependent? | Heterogeneous but inconsistent across pairs — no rule generalises |
| **CausalImpact** [28] | Did *this* event move volatility? | 6/20 ECB meetings significant after FDR (vs. ~5% under noise) |

Four independently authored methods, different assumptions, converging on one variable is stronger evidence than any single p-value; none establishes structural causality — every claim remains temporal-precedence or potential-outcomes under its method's assumptions.

**Two roles, one shipped.** The **validation role is implemented and load-bearing** — it is the actual filter behind the frozen feature set. The **forecast-performance role** (adjusting the forecast directly) was tested and produced **no significant QLIKE improvement after Holm correction**, plausibly because HAR already absorbs the information indirectly; consistent with this thesis's negative-findings policy, it moves to Deferred / Future Work.

### Proposed System Architecture

*(Pipeline and API: implemented, operating daily; extensions below are proposed.)* Five layers share one computation layer across two deployment targets, so the live site and thesis numbers share one code path:

1. **Data pipeline (`collect/`).** One-time historical bootstrap plus a daily job (four times/day via CI) building daily bars, realized measures, and the macro calendar.
2. **Data store (`data/`).** Frozen historical CSVs kept separate from a daily-overwritten live store, so development and sealed segments cannot be silently mixed.
3. **Computation layer (`src/`, `api/cache.py`).** Series join, HAR forecast, calibration, and sizing run in one process, no serialisation boundary.
4. **API (`api/`, FastAPI).** One router per concern (`meta`, `market`, `forecast`, `risk`, `admin`) with Pydantic models, exposing endpoints extended through the Research Plan.
5. **Web application.** One HTML/JS template is the source of truth, compiled to three deployed variants.

**The two deployment targets are a stated trade-off, not an inconsistency.** The local/CI deployment runs the full FastAPI process and is the only place risk numbers are *computed*; Vercel's serverless size limit cannot host that stack, so a daily job pre-computes once-a-day numbers as static JSON while a lightweight function handles only intraday chart data — practice consistent with scaling ML-serving systems under heterogeneous constraints [31], [32].

**Module contract**, enforced in code, not an informal call:

| From | To | Via | Format |
|---|---|---|---|
| `collect/*.py` | `data/` | direct write | fixed columns |
| `data/` | `api/cache.py` | fixed-path read | `Date, open, …, rv5` |
| `src/*.py` | `api/cache.py` | in-process call | DataFrame, no serialisation |
| `api/cache.py` | `api/routers/*.py` | in-memory dict | `pan`, `xs`, `sig`, … |
| `api/routers/*.py` | web client | HTTP JSON | `response_model` |
| daily job | static site data | HTTP call to API | identical JSON, never recomputed |

The daily job never recomputes through a second code path — it calls the API and snapshots the response, the discipline [31] that keeps the static and live deployments from diverging, applied to every module added under the Research Plan.

## Research Methodology

- **Literature review.** HAR family and extensions; FX directional-predictability evidence; statistically sound pattern discovery and multiple-testing control; conformal prediction; VaR/ES backtesting and ruin-constrained sizing.
- **Data preparation and protocol fixing.** Construct bars and realized measures; verify contiguity and session conventions; fix the partition, metrics, multiple-testing procedure, and sealed-set policy in writing **before** any scoring; anti-leakage self-test (truncating information after *t* leaves the value at *t* unchanged).
- **Baseline and model comparison.** Extended HAR and online combination as baselines; compare against regularised linear, tree, recurrent/attention, and foundation models [11], [14] under an identical information set; evaluate with QLIKE [15], CRPS, log/Brier [16]; test with Diebold–Mariano [17], superior predictive ability [5], and Model Confidence Sets [12].
- **Pattern, exogenous, and causal analysis.** Enumerate each discovery family's hypothesis space in advance, step-down max-*T* resampling [4]; condition on the strongest null. The exogenous/causal layer (*Causal Discovery Method*) is complete; remaining work is write-up (Week 9–10) — the forecast overlay stays deferred.
- **Calibration, risk, decision layer.** Mondrian ACI [19], [20]; VaR/ES backtests [21]–[24]; sizing as min(growth-optimal fraction [25], ruin-constrained cap), adjusted by regime, drawdown, correlation, and slippage.
- **Power analysis and sealed validation.** Inject synthetic effects through the actual funnel, report the MDES at 80% power; freeze the configuration, then open the sealed set exactly once.
- **System implementation.** Maintain the operating product path throughout, integrating each layer as a versioned artefact the week it completes; harden the system in the final weeks with full provenance; analyse failure cases and calibration degradation in the interface rather than omitting them.

## Expected Results and Contributions

- **A validated FX decision support system** issuing, per session and pair, a volatility forecast, calibrated three-class probability with a conformal set, event-risk assessment, and ruin-constrained sizing, every quantity traceable to a measurement.
- **An axis-resolved characterisation of predictability** — significant skill on magnitude, none on direction across twelve branches; remaining horizons and the sealed set (not yet opened) complete the picture.
- **A systematic model-complexity benchmark** for volatility forecasting across model families, with proper scoring rules and model confidence sets.
- **A five-method causal triangulation and ablation with an honest two-role split.** Causally-filtered variables degrade less out of sample than validation-selected ones (mechanism: distribution drift in the latter); the triangulation's validation role is load-bearing, its forecast-performance role tested and reported negative rather than overstated.
- **Power-bounded negative evidence** — every negative finding with its MDES, turning "no relationship found" into "no relationship stronger than *X*".
- **Three applied risk-decision contributions**: a portfolio correlation factor for ruin-constrained sizing, a holding-horizon table against stop-out misreading, and a slippage coefficient measured from real executions.
- **A reproducible research record** — data dictionary, sealed-set protocol, full configuration registry, reproduction commands including negative results.
- **A documented, versioned software architecture** — the five-layer pipeline/API/web application with an explicit module contract, so the engineering contribution is itself reported and defensible.

## References

[1] R. A. Meese and K. Rogoff, "Empirical exchange rate models of the seventies: Do they fit out of sample?," *Journal of International Economics*, vol. 14, no. 1–2, pp. 3–24, 1983, doi: 10.1016/0022-1996(83)90017-X.

[2] P.-H. Hsu, M. P. Taylor, and Z. Wang, "Technical trading: Is it still beating the foreign exchange market?," *Journal of International Economics*, vol. 102, pp. 188–208, 2016, doi: 10.1016/j.jinteco.2016.03.012.

[3] M. C. Hutchinson, P. E. Kyziropoulos, J. O'Brien, P. O'Reilly, and T. Sharma, "Technical trading rule profitability in currencies: It's all about momentum," *Research in International Business and Finance*, vol. 63, Art. no. 101779, 2022, doi: 10.1016/j.ribaf.2022.101779.

[4] P. H. Westfall and S. S. Young, *Resampling-Based Multiple Testing*. New York, NY, USA: John Wiley & Sons, 1993.

[5] P. R. Hansen, "A test for superior predictive ability," *Journal of Business & Economic Statistics*, vol. 23, no. 4, pp. 365–380, 2005, doi: 10.1198/073500105000000063.

[6] W. Hämäläinen and G. I. Webb, "A tutorial on statistically sound pattern discovery," *Data Mining and Knowledge Discovery*, vol. 33, no. 2, pp. 325–377, 2019, doi: 10.1007/s10618-018-0590-x.

[7] F. Corsi, "A simple approximate long-memory model of realized volatility," *Journal of Financial Econometrics*, vol. 7, no. 2, pp. 174–196, 2009, doi: 10.1093/jjfinec/nbp001.

[8] T. Bollerslev, A. J. Patton, and R. Quaedvlieg, "Exploiting the errors," *Journal of Econometrics*, vol. 192, no. 1, pp. 1–18, 2016, doi: 10.1016/j.jeconom.2015.10.007.

[9] O. E. Barndorff-Nielsen and N. Shephard, "Power and bipower variation with stochastic volatility and jumps," *Journal of Financial Econometrics*, vol. 2, no. 1, pp. 1–37, 2004, doi: 10.1093/jjfinec/nbh001.

[10] R. R. Branco, A. Rubesam, and M. Zevallos, "Forecasting realized volatility: Does anything beat linear models?," *Journal of Empirical Finance*, vol. 78, Art. no. 101524, 2024, doi: 10.1016/j.jempfin.2024.101524.

[11] A. Brini, "Forecasting realized volatility with time series foundation models," arXiv:2607.05291, 2026.

[12] P. R. Hansen, A. Lunde, and J. M. Nason, "The model confidence set," *Econometrica*, vol. 79, no. 2, pp. 453–497, 2011, doi: 10.3982/ECTA5771.

[13] Y. Freund and R. E. Schapire, "A decision-theoretic generalization of on-line learning," *Journal of Computer and System Sciences*, vol. 55, no. 1, pp. 119–139, 1997, doi: 10.1006/jcss.1997.1504.

[14] N. Hollmann *et al.*, "Accurate predictions on small data with a tabular foundation model," *Nature*, vol. 637, pp. 319–326, 2025, doi: 10.1038/s41586-024-08328-6.

[15] A. J. Patton, "Volatility forecast comparison using imperfect volatility proxies," *Journal of Econometrics*, vol. 160, no. 1, pp. 246–256, 2011, doi: 10.1016/j.jeconom.2010.03.034.

[16] T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," *JASA*, vol. 102, no. 477, pp. 359–378, 2007, doi: 10.1198/016214506000001437.

[17] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995, doi: 10.1080/07350015.1995.10524599.

[18] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, and D. Sejdinovic, "Detecting and quantifying causal associations in large nonlinear time series datasets," *Science Advances*, vol. 5, no. 11, Art. no. eaau4996, 2019, doi: 10.1126/sciadv.aau4996.

[19] I. Gibbs and E. Candès, "Adaptive conformal inference under distribution shift," *NeurIPS 35*, 2021, pp. 1660–1672.

[20] A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle introduction," *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494–591, 2023, doi: 10.1561/2200000101.

[21] P. H. Kupiec, "Techniques for verifying the accuracy of risk measurement models," *The Journal of Derivatives*, vol. 3, no. 2, pp. 73–84, 1995, doi: 10.3905/jod.1995.407942.

[22] P. F. Christoffersen, "Evaluating interval forecasts," *International Economic Review*, vol. 39, no. 4, pp. 841–862, 1998, doi: 10.2307/2527341.

[23] R. F. Engle and S. Manganelli, "CAViaR," *Journal of Business & Economic Statistics*, vol. 22, no. 4, pp. 367–381, 2004, doi: 10.1198/073500104000000370.

[24] A. J. Patton, J. F. Ziegel, and R. Chen, "Dynamic semiparametric models for expected shortfall (and value-at-risk)," *Journal of Econometrics*, vol. 211, no. 2, pp. 388–413, 2019, doi: 10.1016/j.jeconom.2018.10.008.

[25] J. L. Kelly, "A new interpretation of information rate," *The Bell System Technical Journal*, vol. 35, no. 4, pp. 917–926, 1956, doi: 10.1002/j.1538-7305.1956.tb03809.x.

[26] I. Orquín-Serrano, "Predictive power of adaptive candlestick patterns in Forex market," *Mathematics*, vol. 8, no. 5, Art. no. 802, 2020, doi: 10.3390/math8050802.

[27] D. Arnott and G. Pervan, "A critical analysis of decision support systems research," *Journal of Information Technology*, vol. 20, no. 2, pp. 67–87, 2005, doi: 10.1057/palgrave.jit.2000035.

[28] K. H. Brodersen, F. Gallusser, J. Koehler, N. Remy, and S. L. Scott, "Inferring causal impact using Bayesian structural time-series models," *Annals of Applied Statistics*, vol. 9, no. 1, pp. 247–274, 2015, doi: 10.1214/14-AOAS788.

[29] J. Runge, A. Gerhardus, G. Varando, V. Eyring, and G. Camps-Valls, "Causal inference for time series," *Nature Reviews Earth & Environment*, vol. 4, pp. 487–505, 2023, doi: 10.1038/s43017-023-00431-y.

[30] A. N. Angelopoulos, E. J. Candès, and R. J. Tibshirani, "Conformal PID control for time series prediction," *NeurIPS 36*, 2023. arXiv:2307.16895.

[31] S. Shankar, R. Garcia, J. M. Hellerstein, and A. G. Parameswaran, "Operationalizing machine learning: An interview study," arXiv:2209.09125, 2022.

[32] S. Karanam and J. Bhargav, "Microservice architecture patterns for scalable machine learning systems," arXiv:2603.13672, 2026.

[33] V. Chernozhukov *et al.*, "Double/debiased machine learning for treatment and structural parameters," *The Econometrics Journal*, vol. 21, no. 1, pp. C1–C68, 2018, doi: 10.1111/ectj.12097.

[34] S. Wager and S. Athey, "Estimation and inference of heterogeneous treatment effects using random forests," *JASA*, vol. 113, no. 523, pp. 1228–1242, 2018, doi: 10.1080/01621459.2017.1319839.

## Research Plan and Timeline

Research timeline: 16 weeks. Every week pairs a research task with an **Engineering** deliverable, so the API/web application is extended continuously rather than only at the end.

| No. | Assignments | Timeline |
|---|---|---|
| 1 | – Consolidate the pipeline; fix the evaluation protocol; reproduce baselines. <br> – **Eng:** verify `/health`/`/meta` diagnostics; tag the architecture's starting version. | Week 1–2 |
| 2 | – Complete pattern-discovery families at h=5/20; document the hypothesis space. <br> – **Eng:** register results in the `/models` registry. | Week 3–4 |
| 3 | – MDES analysis for every negative finding; validate the funnel with negative controls. <br> – **Eng:** add MDES to `/calibration`. | Week 5–6 |
| 4 | – Address tail-risk failures (two pairs, 99% VaR/ES); recalibrate h=20. <br> – **Eng:** update `/risk` and `/forecast_next` fields. | Week 7–8 |
| 5 | – Write up the completed exogenous/causal analysis, per axis. <br> – **Eng:** expose causal-validation provenance as a read-only `/forecast` field. | Week 9–10 |
| 6 | – End-to-end economic result after costs/slippage; accumulate the forecast journal. <br> – **Eng:** extend `/journal` with rolling calibration. | Week 11 |
| 7 | – Freeze the final configuration in writing; prepare the sealed-set record. <br> – **Eng:** tag and freeze API schemas and the artefact version. | Week 12 |
| 8 | – Open the sealed set once; score without retuning. <br> – **Eng:** serve results through the same frozen endpoints. | Week 13 |
| 9 | – Complete the system: computation layer, versioned artefacts, API, UI with full provenance; verify continuous operation. | Week 14 |
| 10 | – Analyse behaviour by regime/pair/year; document failure cases. <br> – **Eng:** add regime/pair/year breakdowns to the UI. | Week 15 |
| 11 | – Complete the thesis report; revise per feedback; prepare the defence. | Week 16 |

## Deferred / Future Work (Not Scheduled)

- **Causal-conditioned forecast overlay.** The VIX effect is robust (6/6 pairs), but overlaying it on the HAR forecast gave no significant QLIKE gain — deferred pending a better functional form.
- **Contemporaneous causal edges (PCMCI+, τ=0).** Sign-ambiguous edges inconsistent with the frozen lagged result — needs a larger sample or a nonlinear test.
- **Cross-pair volatility spillover (DYNOTEARS).** Real structure detected but does not improve test-set QLIKE as an added feature — not yet forecastable at a daily horizon.
- **FX-specific implied volatility (`EVZCLS`)** was discontinued in March 2025 and is unavailable.

None of the above blocks the sealed-set evaluation or system hardening in Weeks 12–15.

## Work Assignment

*(Percentage split confirmed against actual division of labour, not the undifferentiated 50/50 the advisor rejected. Both members remain jointly responsible for the whole system.)*

| Assignments | Member A (Nguyen Quoc Huy) | Member B (Huynh Tran Quoc Huy) |
|---|---|---|
| Review literature and define research scope | 55% | 45% |
| Construct the data pipeline and realized measures | 40% | 60% |
| Fix the evaluation protocol and sealed-set policy | 50% | 50% |
| Develop and benchmark the volatility forecasting layer | 65% | 35% |
| Develop the pattern-discovery and symbolic-representation branch | 65% | 35% |
| Implement multiple-testing control and stability screening | 60% | 40% |
| Conduct minimum detectable effect size and power analysis | 60% | 40% |
| Develop the exogenous, causal, and cross-asset information layers | 65% | 35% |
| Implement conformal prediction and calibration diagnostics | 40% | 60% |
| Develop the risk layer: VaR/ES backtesting and position sizing | 55% | 45% |
| Conduct the end-to-end economic evaluation | 50% | 50% |
| Execute the sealed-set validation | 50% | 50% |
| Design and implement the data pipeline scheduling and module contracts | 35% | 65% |
| Implement the API layer (FastAPI routers, response schemas) | 30% | 70% |
| Implement the web application and its three deployment variants | 30% | 70% |
| Analyse failure cases and document limitations | 50% | 50% |
| Write and revise the thesis report | 50% | 50% |
| Prepare for the final defence | 50% | 50% |

<!-- KHOI-KY -->
<!-- Bo sinh .docx (src/xuat_proposal.py) tu dung o ky hai cot tu day.
     Khong viet bang &nbsp; nua — no khong len duoc .docx cho dep. -->
