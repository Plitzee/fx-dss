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

The foreign exchange (FX) market is the largest and most liquid financial market in the world, and many decision-support tools marketed to retail participants display a directional forecast, chart-pattern signals, and a qualitative confidence label whose empirical foundation is weaker than their presentation suggests. Daily exchange-rate returns have resisted prediction since Meese and Rogoff [1], and large-scale replications of technical trading rules report that in-sample profitability largely disappears out of sample, with residual abnormal returns absorbed by a time-series momentum factor [2], [3]. The central difficulty is inference under a low signal-to-noise ratio combined with a large hypothesis space: a system that evaluates thousands of candidate rules will find many spurious relationships unless the hypothesis space is enumerated in advance and the family-wise error rate is controlled [4]–[6]. A related, less-addressed difficulty is that a negative finding is uninformative without its minimum detectable effect size (MDES) — otherwise "there is nothing there" and "the test could not have found it" are indistinguishable. Predictability is also not one property: the *direction* and *magnitude* axes have different empirical status — the HAR model of realized volatility [7]–[9] gives well-established magnitude forecasts while directional skill remains contested [10], [11] — so a decision support system should state explicitly which axis it claims skill on and demonstrate that under a fixed protocol.

Preliminary work by the group has established feasibility through three staged, pre-registered phases, each closed with a written decision record. A complete data pipeline covers six major pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF) over 2010–2025 (≈34.9 million one-minute bars); a production volatility layer, a three-class probability layer, and a risk layer run end to end through an operating web application updated on a daily schedule. The extended HAR specification attains a test-set QLIKE of 0.1585 versus 0.2172 for the prior baseline, and fourteen ML/DL alternatives plus 509 forecast combinations fail to displace it from the Model Confidence Set [12]. On the direction axis, 8,652 enumerated rule hypotheses across twelve discovery branches (price patterns and central-bank communication) yield zero survivors after family-wise error control, with measured detection power at a lift of 1.20. A further pre-registered ablation of exogenous macro/cross-asset variables under a causal-versus-validation feature-selection design is completed and frozen; its sealed-set evaluation was withheld because the pre-specified opening rule was not met.

This thesis therefore addresses a problem that is as much **engineering** as it is statistical: constructing a decision support system whose every displayed number is traceable to a pre-registered measurement, states the axis on which it has skill, quantifies its negative findings, and is delivered as software that runs continuously rather than as a script for a report. The data pipeline, computation layer, API, and web application (Proposed Method and System Architecture) are a primary deliverable, developed across the full sixteen weeks rather than assembled at the end. The remaining research work is a small number of explicitly identified residual validation items, the single sealed-set evaluation, and system hardening.

## Market Survey

The group surveyed existing retail FX decision-support tools so the contribution below is stated relative to what exists, not in the abstract. **Signal marketplaces** (MQL5/MetaTrader, TradingView public scripts) publish a track record from the vendor's own in-sample selection, with no pre-registered protocol or multiple-testing correction across the many scripts a user browses. **Portfolio-analytics services** (Myfxbook, FX Blue) report realized trade statistics but no forward-looking calibrated probabilities or backtested risk figures. **Broker "confidence"/"sentiment" widgets** attach a qualitative label with no proper scoring rule or calibration diagnostic. None discloses the hypothesis space searched, the multiple-testing correction applied, a calibration guarantee, or a quantified probability of ruin — properties missing from applied decision support systems generally [27]. This gap, not a claim of superior directional accuracy, is what the proposed system targets.

## Research Objectives

The primary objective is to design, implement, and empirically validate a decision support system for FX trading in which forecast skill is established per target axis under strict multiple-testing control, uncertainty is communicated through calibrated conformal prediction, and position sizing is constrained by an estimated probability of ruin.

Three groups of quantities are kept conceptually distinct throughout: **predictive axes** (direction, magnitude, tail-event probability, each scored against its own baseline), **calibration diagnostics** (coverage and reliability), and the **decision layer** (VaR, ES, stop-out probability, position sizing).

Specific objectives:

- **Establishing axis-specific predictability.** Determine, under one fixed protocol, on which axis — direction, magnitude, or risk — out-of-sample skill exists for daily FX, reported separately per axis.
- **Benchmarking model complexity against a well-specified econometric baseline.** Compare linear HAR-type specifications with tree-based, recurrent, attention-based, and foundation-model alternatives, and systematic forecast combinations, under proper scoring rules and formal comparison tests.
- **Quantifying the incremental value of additional information layers.** Evaluate whether central-bank text, exogenous macro/cross-asset variables, and candlestick geometry [26] add predictive value, and whether causal filtering outperforms ordinary feature selection.
- **Delivering calibrated uncertainty.** Implement and validate adaptive conformal prediction for target coverage under distribution shift, verified empirically and by regime, with proper scoring rules and reliability diagnostics.
- **Converting forecasts into risk-controlled decisions.** Produce a per-session decision record: forecast volatility, stop distance, stop-out probability by horizon, VaR/ES with formal backtests, and a ruin-constrained position size adjusted for portfolio correlation and slippage.
- **Reporting negative findings with quantified power.** Report the MDES for every negative result, so each negative claim is bounded rather than open-ended.
- **Engineering and operating a production decision support architecture.** Design, implement, and extend — throughout the sixteen weeks rather than only at the end — a five-layer software architecture (data pipeline, computation layer, versioned forecast artefacts, a documented API, and a web application) so every research layer is integrated as soon as it is completed and every displayed number stays traceable to its measurement (Proposed Method and System Architecture).

## Research Scope

This research focuses on **daily-horizon decision support for six major USD pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF. The development sample (2010–2025) is split into training, validation (model/hyper-parameter selection only), and a held-out development-test segment scored once, distinct from the sealed set below.

A **sealed evaluation set** — six cross pairs never used in development (EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY, EUR/CHF, NZD/USD) plus 2026 data for all twelve pairs as available at a pre-specified cutoff — is opened exactly once after the final configuration is frozen in writing, with no retuning afterward. Because the sealed set includes calendar year 2026 while the study concludes before that year ends, "2026" means whatever the continuously operating daily pipeline has collected by the cutoff, not the completed year.

The primary source is one-minute historical bar data, from which daily bars and five-minute realized measures (variance, quarticity, bipower variation, signed semivariances) are computed. Supplementary sources: the central-bank/macro release calendar, daily exogenous series from a public economic data service, monetary-policy surprise measures, interest-rate differentials, and tick-derived transaction-cost measurements.

Forecast horizons are one, five, and twenty sessions, with one session the operational default; longer horizons are reported with their measured skill rather than presented as equivalent. The study explicitly does **not** attempt a profitable automated trading strategy — order execution, microstructure, market impact, sub-daily timing, and cross-asset portfolio optimisation are out of scope. Causality claims are limited to temporal predictive relationships in the Granger sense, not structural causal claims.

## Research Subjects

The primary research subject is **the statistical validation of decision support outputs for foreign exchange trading under a low signal-to-noise ratio**.

The study examines how forecast skill is attributed to a target axis and an information layer; how multiple-testing control, power analysis, leakage protection, and sealed-set validation combine to produce claims that survive replication; how calibrated forecasts and conformal sets convert into risk limits with a measured probability of ruin; and how this evidence should be presented so a user can distinguish what the system measures from what it merely assumes.

## Proposed Method and System Architecture

This section states what is proposed, labelling each item **(implemented)** or **(proposed)** to distinguish components already frozen from remaining work; nothing here is presented as finished until it appears with measured numbers in Expected Results or the defended thesis.

### Proposed Predictive and Risk Layers

- **Magnitude model — extended HAR.** *(implemented)* A heterogeneous-autoregressive specification on realized variance [7], extended with measurement-error correction [8] and jump/semivariance decomposition [9], selected over fourteen ML/DL challengers and 509 combinations under the Model Confidence Set [12].
- **Direction/probability model — online expert combination.** *(implemented)* A Hedge-style combination [13] of four sub-models (climatology, persistence, σ̂-only, σ̂+regime), re-weighted every session; challenged by regularised linear, tree, recurrent, attention, and foundation-model alternatives.
- **Causal discovery module.** *(implemented — validation role; forecast-integration tested and deferred)* Five triangulated methods decide which exogenous variable is trusted; using the estimated effect to adjust the forecast directly was tested and found not yet effective (below; Deferred / Future Work).
- **Calibration — Mondrian-stratified adaptive conformal inference.** *(implemented)* ACI [19] stratified by volatility regime [20], 1.2-point maximum calibration deviation versus 2.4–3.2 for four alternatives. *(proposed)* A PID-controller-based online update [30] is evaluated as a candidate refinement for the below-target coverage observed during drawdown.
- **Risk/decision module.** *(implemented)* Empirical-quantile VaR/ES with Kupiec [21], Christoffersen [22], and dynamic-quantile [23] backtests and a consistent joint scoring function [24]; Kelly-derived position size [25] capped by an estimated probability of ruin. *(proposed)* Portfolio-correlation adjustment and a slippage coefficient measured from observed stop-loss executions.

### Causal Discovery Method

The exogenous layer needs a method that conditions on many candidates simultaneously rather than pairwise, separates a direct link from a shared-confound artefact, and plugs into the same false-discovery control as the price-pattern families. The **official decision gate** for the frozen exogenous feature-selection ablation is Granger-type nested-regression F-testing under step-down Westfall–Young control [4]. Beyond it, four independent, non-self-written methods triangulate the result — this is the answer to "which algorithm, and how do you know it is real causality rather than random correlation":

| Method | Question answered | Key result |
|---|---|---|
| **PCMCI** [18], [29] (`tigramite`) | Does the link survive simultaneous conditioning on the other candidates? | VIX survives on every pair tested; gold-implied volatility (GVZCLS) looks causal alone but loses significance once conditioned on VIX — the concrete confound-vs-cause distinction pairwise Granger cannot make |
| **Double Machine Learning** [33] (`doubleml`) | What is the effect *size*, with a nonlinear nuisance model? | VIX effect positive, Holm-significant in **6/6** pairs (θ = 0.070–0.186) — the strongest, most consistent result in the triangulation |
| **Causal Forest** [34] (`econml`) | Is the effect regime-dependent? | Heterogeneous but not consistent across pairs (4/6 increase with stress, 2 do not) — no single rule generalises |
| **CausalImpact** [28] (`pycausalimpact`) | Did *this* specific event move volatility past its counterfactual? | Piloted on 20 ECB meetings: 6/20 significant after FDR correction (vs. ~5% expected under noise), all positive |

Four independently authored methods, resting on different assumptions (linear/nonlinear, pairwise/simultaneous, existence/magnitude/heterogeneity/single-event), converging on the same variable is stronger evidence than any single p-value — this triangulation, not one algorithm, is how the causality-versus-correlation question is answered. Consistent with the Research Scope, none of the five methods is presented as establishing structural causality; every statement remains a temporal-precedence or potential-outcomes claim under its own method's assumptions.

**Two roles, only one shipped.** The triangulation's **validation role is implemented and load-bearing**: it is the actual filter that kept VIX and dropped ten other candidates from the frozen feature set. Its **forecast-performance role — using the effect to directly adjust the production forecast (constant and regime-conditional overlays) — produced no significant QLIKE improvement in any pair after Holm correction**, most plausibly because the HAR baseline already absorbs the information indirectly. Consistent with this thesis's negative-findings policy, this application is not deployed and moves to Deferred / Future Work rather than the remaining timeline.

### Proposed System Architecture

*(Data pipeline and API: implemented and operating on a daily schedule; extensions in the Research Plan are proposed.)*

Five layers share one computation layer across two deployment targets, so the live site and the numbers computed for the thesis come from the same code path:

1. **Data pipeline (`collect/`).** A one-time historical bootstrap plus a daily job (`live_fx.py`, `lich_su_kien.py`, run four times a day via CI) build daily bars, five-minute realized measures, and the macro/central-bank calendar.
2. **Data store (`data/`).** Frozen historical CSVs are kept separate from a daily-overwritten live store, so development and sealed-evaluation segments cannot be silently mixed by a pipeline bug.
3. **Computation layer (`src/`, `api/cache.py`).** The series join, HAR forecast, three-class combination, conformal calibration, and risk/sizing functions run in one process, with no serialisation boundary to the layer that serves them.
4. **API (`api/`, FastAPI).** One router per concern (`meta`, `market`, `forecast`, `risk`, `admin`) with Pydantic response models, already exposing the endpoints extended through the Research Plan below (`/health`, `/meta`, `/series`, `/indicators`, `/forecast`, `/forecast_series`, `/forecast_next`, `/journal`, `/calibration`, `/models`, `/risk`, `/events`, `/refresh`).
5. **Web application.** One HTML/JS template (`web/ui_template.html`) is the source of truth, compiled to three deployed variants rather than maintained as three hand-written pages.

**The two deployment targets are a stated engineering trade-off, not an inconsistency — this answers how the modules are linked at the deployment level.** The local/CI deployment runs the full FastAPI process and is the only place risk numbers are *computed*; Vercel's serverless size limit cannot host that dependency stack, so the daily job pre-computes the once-a-day numbers as static JSON while a separate lightweight function computes only the intraday chart data. Concentrating computation in one synchronous service that a lighter target consumes as a versioned artefact follows established practice for scaling ML-serving systems under heterogeneous deployment constraints [31], [32].

**Module contract — the mechanism, not just the diagram.** Each row is a fixed data contract enforced in code, not an informal call:

| From | To | Via | Format |
|---|---|---|---|
| `collect/*.py` | `data/` | direct CSV/JSON write | fixed columns, no silent schema change |
| `data/` | `api/cache.py` | fixed-path CSV read | `Date, open, high, low, close, rv5, …` |
| `src/*.py` | `api/cache.py` | in-process function call | numpy/DataFrame, no serialisation |
| `api/cache.py` | `api/routers/*.py` | in-memory dictionary | `pan`, `xs`, `sig`, `che_do`, … |
| `api/routers/*.py` | web client | HTTP JSON | Pydantic `response_model` where declared |
| daily job | static site data | HTTP call to the API | identical JSON — never recomputed by a second formula |

The daily job never recomputes a number through a second code path — it calls the API and snapshots the response. This single-source-of-truth discipline, recommended practice for operationalising ML systems reliably [31], is what keeps the static and live deployments from ever diverging, and applies to every module added under the Research Plan below, not only the modules that exist today.

## Research Methodology

The research proceeds through the following activities.

### Literature Review
- Realized-volatility modelling: the HAR family and its measurement-error, jump, and semivariance extensions; recent ML/foundation-model comparisons.
- FX directional predictability: large-scale technical-rule studies with data-snooping control; the momentum-factor explanation.
- Statistically sound pattern discovery, family-wise error control, superior predictive ability testing, model confidence sets.
- Conformal prediction under distribution shift, proper scoring rules, calibration diagnostics.
- VaR/ES backtesting and growth-optimal position sizing under a ruin constraint.

### Data Preparation and Protocol Fixing
- Construct daily bars and five-minute realized measures from one-minute data; verify contiguity, session conventions, and time-zone handling.
- Fix the temporal partition, per-axis metrics, multiple-testing procedure, and sealed-set policy in writing **before** any scoring; record every configuration evaluated.
- Anti-leakage self-test: every derived feature's value at time *t* is unchanged by truncating information revealed after *t*.

### Baseline Establishment and Model Comparison
- Extended HAR as the production magnitude baseline; an online expert-combination layer [13] as the probability baseline.
- Compare against regularised linear models, gradient-boosted trees, recurrent/attention networks, a tabular foundation model [14], zero-shot time-series foundation models [11], and forecast combinations, under an identical information set and refit frequency.
- Evaluate with scale-invariant QLIKE [15], CRPS, and log/Brier scores [16]; assess significance with Diebold–Mariano tests [17] and Model Confidence Sets [12].

### Pattern, Exogenous, and Causal Analysis
- Enumerate the hypothesis space in advance per discovery family and apply step-down max-*T* resampling with block-preserving nulls [4]; remaining work is limited to pre-registered evaluations at horizons/axes not yet scored — no new family is introduced after the protocol is frozen.
- Condition every candidate on the strongest available null (production volatility forecast for magnitude; time-series momentum for direction [3]).
- For the exogenous layer, contrast all-variable, validation-selected, and causally-filtered feature sets (Granger+Westfall–Young, triangulated against PCMCI [18], Double Machine Learning [33], Causal Forest [34], and CausalImpact [28] — Proposed Method, Causal Discovery Method) under an identical model class and parameter budget. This layer, including the triangulation, is complete; remaining work is limited to writing up the findings (Week 9–10), and the causal-conditioned forecast overlay is deferred (Deferred / Future Work) rather than pursued further here.
- Screen for sign consistency across pairs and years, and require leave-one-pair-out transfer before any candidate is admitted.

### Calibration, Risk, and Decision Layer
- Adaptive conformal inference with Mondrian stratification by volatility regime [19], [20]; verify empirical coverage overall and by regime.
- Estimate VaR/ES from empirical quantiles of standardised returns; backtest per pair with unconditional [21], conditional [22], and dynamic-quantile [23] tests; assess the joint (VaR, ES) pair with a strictly consistent scoring function [24].
- Compute position size as the minimum of a growth-optimal fraction [25] and a ruin-constrained cap, adjusted by regime, drawdown, portfolio correlation, and a measured slippage coefficient.

### Power Analysis and Sealed Validation
- For every negative finding, inject synthetic effects through the actual testing funnel and report the effect size detected at 80% power, with a negative control for the false-positive rate under pure noise.
- Freeze the final configuration in writing, then open the sealed evaluation set exactly once and report the resulting numbers without modification.

### System Implementation and Analysis
- Maintain the operating end-to-end product path (Proposed Method and System Architecture) throughout the study, integrating each research layer as a versioned artefact the week it is completed (per-week deliverables in the Research Plan); harden the system in the final weeks with full provenance for every displayed quantity.
- Analyse failure cases and limitations — pairs failing tail backtests, horizons where calibration degrades — and report them within the interface rather than omitting them.

## Expected Results and Contributions

- **A validated FX decision support system** issuing, per session and pair, a volatility forecast, a calibrated three-class probability with a conformal prediction set, an event-risk assessment, and a ruin-constrained position-sizing recommendation, with every displayed quantity traceable to a specific measurement.
- **An axis-resolved characterisation of daily FX predictability.** Results frozen to date show positive, statistically significant skill on the magnitude axis and none on direction across twelve independent discovery branches; the thesis completes the remaining horizons and confirms this on the sealed evaluation set, not yet opened.
- **A systematic benchmark of model complexity for FX volatility forecasting** — linear, tree-based, recurrent, attention-based, and foundation models, plus exhaustive forecast combinations, evaluated with proper scoring rules and model confidence sets.
- **An empirical comparison of causal filtering against ordinary feature selection, and a five-method causal triangulation with an honest split between its two roles.** A completed, frozen ablation — its sealed-set evaluation deliberately not opened, per the pre-registered rule that a non-positive verdict does not warrant it — shows causally-filtered variables degrade substantially less out of sample than validation-selected ones, the mechanism measured as distribution drift in the latter. Separately, Granger+Westfall–Young, PCMCI, Double ML, Causal Forest, and CausalImpact independently converge on VIX and demonstrate concretely (PCMCI's simultaneous conditioning) how it is distinguished from a confound. The triangulation's validation role (deciding which variable is trusted) is complete and load-bearing; its forecast-performance role was tested and found not yet effective, reported as a negative finding rather than an overstated gain.
- **Power-bounded negative evidence** — every negative finding reported with the minimum effect size its procedure could detect, converting "no relationship was found" into "no relationship stronger than *X* exists in these data".
- **Three applied contributions to risk-aware decision support**: a portfolio correlation factor quantifying the gap between nominal and realised ruin probability under correlated positions; a holding-horizon table addressing systematic misreading of single-session stop-out probabilities; and a slippage coefficient estimated from observed stop-loss executions rather than assumed.
- **A reproducible research record** — the full data dictionary, sealed-set protocol, a registry of every configuration and hypothesis evaluated, and reproduction commands for every reported number, including negative results.
- **A documented, versioned software architecture** — the five-layer pipeline, computation layer, API, and web application with an explicit module contract between every layer, so the engineering contribution (how numbers are produced, versioned, and served without the static and live deployments diverging) is itself reported and defensible, not only the numbers it serves.

## References

[1] R. A. Meese and K. Rogoff, "Empirical exchange rate models of the seventies: Do they fit out of sample?," *Journal of International Economics*, vol. 14, no. 1–2, pp. 3–24, 1983, doi: 10.1016/0022-1996(83)90017-X.

[2] P.-H. Hsu, M. P. Taylor, and Z. Wang, "Technical trading: Is it still beating the foreign exchange market?," *Journal of International Economics*, vol. 102, pp. 188–208, 2016, doi: 10.1016/j.jinteco.2016.03.012.

[3] M. C. Hutchinson, P. E. Kyziropoulos, J. O'Brien, P. O'Reilly, and T. Sharma, "Technical trading rule profitability in currencies: It's all about momentum," *Research in International Business and Finance*, vol. 63, Art. no. 101779, 2022, doi: 10.1016/j.ribaf.2022.101779.

[4] P. H. Westfall and S. S. Young, *Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment*. New York, NY, USA: John Wiley & Sons, 1993.

[5] P. R. Hansen, "A test for superior predictive ability," *Journal of Business & Economic Statistics*, vol. 23, no. 4, pp. 365–380, 2005, doi: 10.1198/073500105000000063.

[6] W. Hämäläinen and G. I. Webb, "A tutorial on statistically sound pattern discovery," *Data Mining and Knowledge Discovery*, vol. 33, no. 2, pp. 325–377, 2019, doi: 10.1007/s10618-018-0590-x.

[7] F. Corsi, "A simple approximate long-memory model of realized volatility," *Journal of Financial Econometrics*, vol. 7, no. 2, pp. 174–196, 2009, doi: 10.1093/jjfinec/nbp001.

[8] T. Bollerslev, A. J. Patton, and R. Quaedvlieg, "Exploiting the errors: A simple approach for improved volatility forecasting," *Journal of Econometrics*, vol. 192, no. 1, pp. 1–18, 2016, doi: 10.1016/j.jeconom.2015.10.007.

[9] O. E. Barndorff-Nielsen and N. Shephard, "Power and bipower variation with stochastic volatility and jumps," *Journal of Financial Econometrics*, vol. 2, no. 1, pp. 1–37, 2004, doi: 10.1093/jjfinec/nbh001.

[10] R. R. Branco, A. Rubesam, and M. Zevallos, "Forecasting realized volatility: Does anything beat linear models?," *Journal of Empirical Finance*, vol. 78, Art. no. 101524, 2024, doi: 10.1016/j.jempfin.2024.101524.

[11] A. Brini, "Forecasting realized volatility with time series foundation models: A comparison with econometric benchmarks," arXiv:2607.05291, 2026. [Online]. Available: https://arxiv.org/abs/2607.05291

[12] P. R. Hansen, A. Lunde, and J. M. Nason, "The model confidence set," *Econometrica*, vol. 79, no. 2, pp. 453–497, 2011, doi: 10.3982/ECTA5771.

[13] Y. Freund and R. E. Schapire, "A decision-theoretic generalization of on-line learning and an application to boosting," *Journal of Computer and System Sciences*, vol. 55, no. 1, pp. 119–139, 1997, doi: 10.1006/jcss.1997.1504.

[14] N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, and F. Hutter, "Accurate predictions on small data with a tabular foundation model," *Nature*, vol. 637, pp. 319–326, 2025, doi: 10.1038/s41586-024-08328-6.

[15] A. J. Patton, "Volatility forecast comparison using imperfect volatility proxies," *Journal of Econometrics*, vol. 160, no. 1, pp. 246–256, 2011, doi: 10.1016/j.jeconom.2010.03.034.

[16] T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359–378, 2007, doi: 10.1198/016214506000001437.

[17] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995, doi: 10.1080/07350015.1995.10524599.

[18] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, and D. Sejdinovic, "Detecting and quantifying causal associations in large nonlinear time series datasets," *Science Advances*, vol. 5, no. 11, Art. no. eaau4996, 2019, doi: 10.1126/sciadv.aau4996.

[19] I. Gibbs and E. Candès, "Adaptive conformal inference under distribution shift," in *Proc. 35th Conf. Neural Information Processing Systems (NeurIPS)*, 2021, pp. 1660–1672.

[20] A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle introduction," *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494–591, 2023, doi: 10.1561/2200000101.

[21] P. H. Kupiec, "Techniques for verifying the accuracy of risk measurement models," *The Journal of Derivatives*, vol. 3, no. 2, pp. 73–84, 1995, doi: 10.3905/jod.1995.407942.

[22] P. F. Christoffersen, "Evaluating interval forecasts," *International Economic Review*, vol. 39, no. 4, pp. 841–862, 1998, doi: 10.2307/2527341.

[23] R. F. Engle and S. Manganelli, "CAViaR: Conditional autoregressive value at risk by regression quantiles," *Journal of Business & Economic Statistics*, vol. 22, no. 4, pp. 367–381, 2004, doi: 10.1198/073500104000000370.

[24] A. J. Patton, J. F. Ziegel, and R. Chen, "Dynamic semiparametric models for expected shortfall (and value-at-risk)," *Journal of Econometrics*, vol. 211, no. 2, pp. 388–413, 2019, doi: 10.1016/j.jeconom.2018.10.008.

[25] J. L. Kelly, "A new interpretation of information rate," *The Bell System Technical Journal*, vol. 35, no. 4, pp. 917–926, 1956, doi: 10.1002/j.1538-7305.1956.tb03809.x.

[26] I. Orquín-Serrano, "Predictive power of adaptive candlestick patterns in Forex market. EURUSD case," *Mathematics*, vol. 8, no. 5, Art. no. 802, 2020, doi: 10.3390/math8050802.

[27] D. Arnott and G. Pervan, "A critical analysis of decision support systems research," *Journal of Information Technology*, vol. 20, no. 2, pp. 67–87, 2005, doi: 10.1057/palgrave.jit.2000035.

[28] K. H. Brodersen, F. Gallusser, J. Koehler, N. Remy, and S. L. Scott, "Inferring causal impact using Bayesian structural time-series models," *Annals of Applied Statistics*, vol. 9, no. 1, pp. 247–274, 2015, doi: 10.1214/14-AOAS788.

[29] J. Runge, A. Gerhardus, G. Varando, V. Eyring, and G. Camps-Valls, "Causal inference for time series," *Nature Reviews Earth & Environment*, vol. 4, pp. 487–505, 2023, doi: 10.1038/s43017-023-00431-y.

[30] A. N. Angelopoulos, E. J. Candès, and R. J. Tibshirani, "Conformal PID control for time series prediction," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023. arXiv:2307.16895.

[31] S. Shankar, R. Garcia, J. M. Hellerstein, and A. G. Parameswaran, "Operationalizing machine learning: An interview study," arXiv:2209.09125, 2022.

[32] S. Karanam and J. Bhargav, "Microservice architecture patterns for scalable machine learning systems," arXiv:2603.13672, 2026.

[33] V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins, "Double/debiased machine learning for treatment and structural parameters," *The Econometrics Journal*, vol. 21, no. 1, pp. C1–C68, 2018, doi: 10.1111/ectj.12097.

[34] S. Wager and S. Athey, "Estimation and inference of heterogeneous treatment effects using random forests," *Journal of the American Statistical Association*, vol. 113, no. 523, pp. 1228–1242, 2018, doi: 10.1080/01621459.2017.1319839.

## Research Plan and Timeline

Research timeline: 16 weeks.

| No. | Assignments | Timeline |
|---|---|---|
| 1 | – Consolidate the data pipeline; verify contiguity, session conventions, and time-zone handling. <br> – Fix the evaluation protocol: partition, per-axis metrics, multiple-testing procedure, sealed-set policy. <br> – Reproduce all baseline results from a clean checkout. <br> – **Engineering:** verify the daily pipeline job and `/health`/`/meta` diagnostics against the reproduced baseline; tag this state as the architecture's starting version. | Week 1–2 |
| 2 | – Complete the pattern-discovery families at the five- and twenty-session horizons. <br> – Apply superior predictive ability testing at the family level. <br> – Document the enumerated hypothesis space in full. <br> – **Engineering:** register the new discovery-family results in the model registry served by `/models`. | Week 3–4 |
| 3 | – Conduct minimum detectable effect size analysis for every negative finding. <br> – Validate the discovery funnel with negative controls and known-effect injection. <br> – Formalise the power-bounded reporting template. <br> – **Engineering:** extend `/calibration` with the MDES figure for every negative finding. | Week 5–6 |
| 4 | – Address tail-risk failures for the two pairs failing VaR/ES backtests at the 99% level, using conditional and extreme-value approaches. <br> – Recalibrate the twenty-session horizon. <br> – **Engineering:** update `/risk`'s VaR/ES fields for the affected pairs and `/forecast_next`'s twenty-session horizon. | Week 7–8 |
| 5 | – Write up the completed exogenous/causal analysis: the ablation and the five-method triangulation (already done, not run here). <br> – Report results separately per axis, as already measured. <br> – **Engineering:** expose causal-validation provenance as a read-only `/forecast` diagnostic field — not used to alter the forecast value. | Week 9–10 |
| 6 | – Evaluate the end-to-end economic result after measured transaction costs and slippage. <br> – Accumulate the forecast journal; compute rolling calibration. <br> – **Engineering:** extend `/journal` with cost-adjusted performance and rolling calibration diagnostics. | Week 11 |
| 7 | – Freeze the final configuration in writing: variables, selection method, models, metrics, decision rules. <br> – Prepare the sealed-set evaluation record. <br> – **Engineering:** tag and freeze the API response schemas and forecast-artefact version. | Week 12 |
| 8 | – Open the sealed evaluation set exactly once; score the frozen configuration. <br> – Report results without modification or retuning. <br> – **Engineering:** serve sealed-set results through the same frozen endpoints used throughout — no separate reporting path. | Week 13 |
| 9 | – Complete the system: computation layer, versioned forecast artefacts, API, and UI with full provenance for every displayed quantity. <br> – Verify continuous daily operation. | Week 14 |
| 10 | – Analyse model behaviour by regime, pair, and year; examine failure cases and document limitations. <br> – **Engineering:** add regime-, pair-, and year-conditional breakdowns to the risk and calibration views. | Week 15 |
| 11 | – Complete the thesis report; revise content per advisor feedback; prepare for the final defence. | Week 16 |

Software-architecture work therefore runs across Weeks 1–15, extending the operating API and web application one layer at a time as each research result is frozen, rather than being concentrated at the end.

## Deferred / Future Work (Not Scheduled in This Thesis)

Directions surfaced by completed experiments that are explicitly **not** part of the sixteen-week plan, kept separate from Proposed Method so the timeline stays honest about what is actually committed:

- **Causal-conditioned forecast overlay.** The VIX effect is real and robust (Double ML, 6/6 pairs significant after Holm correction), but a constant or Causal-Forest regime-conditional overlay on the HAR forecast gave no significant QLIKE improvement in any pair. Deferred pending a better functional form, or acceptance that the baseline already absorbs it.
- **Contemporaneous causal edges (PCMCI+, τ = 0).** Undirected or sign-ambiguous edges between implied-volatility variables and FX volatility, inconsistent with the frozen lagged result — resolving this needs a larger sample or a nonlinear conditional-independence test.
- **Cross-pair volatility spillover (DYNOTEARS).** A self-implemented DYNOTEARS network detects contemporaneous spillover (e.g., AUD/USD↔USD/CAD, EUR/USD→USD/CHF) consistent with an independently run linear Diebold–Yilmaz measure, but neither improves test-set QLIKE when added as an extra HAR feature (six-pair average 0.1281 vs. 0.1285) — real structure, not yet forecastable at a daily horizon.
- **FX-specific implied volatility.** `EVZCLS`, the single most promising untested candidate, was discontinued by CBOE in March 2025 and is unavailable for this sample.

None of the above blocks the sealed-set evaluation or system hardening in Weeks 12–15: the production magnitude forecast continues to rely on the extended-HAR baseline alone, with the causal layer contributing to feature-trust validation but not to the forecast value itself.

## Work Assignment

*(Percentage split confirmed against the two members' actual division of labour, not the undifferentiated 50/50 the advisor explicitly rejected. Both members remain jointly responsible for the correctness of the full system regardless of the split below.)*

| Assignments | Member A (Nguyen Quoc Huy) | Member B (Huynh Tran Quoc Huy) |
|---|---|---|
| Review literature and define research scope | 55% | 45% |
| Construct the data pipeline and realized measures | 40% | 60% |
| Fix the evaluation protocol and sealed-set policy | 50% | 50% |
| Develop and benchmark the volatility forecasting layer | 65% | 35% |
| Develop the pattern-discovery and symbolic-representation branch | 65% | 35% |
| Implement multiple-testing control and stability screening | 60% | 40% |
| Conduct minimum detectable effect size and power analysis | 60% | 40% |
| Develop the exogenous, causal (five-method triangulation), and cross-asset information layers | 65% | 35% |
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
