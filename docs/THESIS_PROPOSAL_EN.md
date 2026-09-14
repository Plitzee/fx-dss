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

The foreign exchange (FX) market is the largest and most liquid financial market in the world, and a large number of decision support tools are marketed to retail participants. Such tools typically display a directional forecast, a set of chart-pattern signals, and a qualitative confidence label. The empirical foundation for these outputs is, however, considerably weaker than their presentation suggests. Daily exchange rate returns have resisted prediction since the benchmark result of Meese and Rogoff [1], and large-scale replications of technical trading rules in currencies report that in-sample profitability largely disappears out of sample and that residual abnormal returns are absorbed by a single time-series momentum factor [2], [3].

The central methodological difficulty is not model capacity but **inference under a low signal-to-noise ratio combined with a large hypothesis space**. A system that evaluates thousands of candidate rules, feature sets, or model configurations will identify many nominally significant relationships purely by chance. Unless the hypothesis space is enumerated in advance and the family-wise error rate is controlled — for example by resampling-based step-down procedures [4] or by tests of superior predictive ability over an entire model family [5] — the resulting system reports discoveries that do not survive replication [6]. A second and less frequently addressed difficulty concerns **negative findings**: reporting that no effect was detected is uninformative unless the minimum detectable effect size (MDES) of the testing procedure is also reported, because "there is nothing there" and "the procedure could not have found it" are otherwise indistinguishable.

A third observation motivates the design adopted here. Predictability in FX is not a single property: it should be decomposed by **target axis**. The *direction* axis (will the price rise or fall) and the *magnitude* axis (how large will the movement be) have very different empirical status. The heterogeneous autoregressive (HAR) model of realized volatility [7] and its extensions incorporating measurement-error correction [8] and jump/semivariance components [9] provide well-established magnitude forecasts, whereas directional skill remains contested. Recent comparative studies further report that nonlinear machine-learning models do not statistically outperform linear specifications for realized volatility once multiple testing is accounted for [10], and that time-series foundation models do not consistently beat a log-HAR benchmark [11]. A decision support system should therefore state explicitly *on which axis* it claims skill, and should demonstrate that claim under a fixed evaluation protocol rather than asserting it.

Preliminary work carried out by the group has established the feasibility and the empirical baseline for this design, and has already progressed through three staged, pre-registered information layers, each closed with a written decision record before the next was opened. A complete data pipeline has been constructed for six major currency pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF) over 2010–2025, aggregating approximately 34.9 million one-minute bars into daily bars and five-minute realized measures; a production volatility layer, a three-class probability layer, and a risk layer have been implemented end to end and are served through an operating web application updated on a daily schedule. Under a fixed 70/15/15 temporal split, the extended HAR specification attains a test-set QLIKE of 0.1585 against 0.2172 for the previous baseline, while fourteen machine-learning and deep-learning alternatives, two zero-shot time-series foundation models, and 509 forecast combinations fail to displace it from the Model Confidence Set [12]. On the direction axis, 8,652 fully enumerated rule hypotheses across twelve independent discovery branches — spanning historical price patterns and, in a subsequent closed phase, four independent representations of official central-bank communication — yield zero survivors after family-wise error control, with a measured detection power corresponding to a lift of 1.20. A further pre-registered investigation of exogenous macroeconomic and cross-asset variables under a causal-versus-validation feature-selection ablation has also been completed and frozen, with its sealed-set evaluation deliberately withheld because the pre-specified decision rule for opening it was not met.

This thesis therefore addresses a research problem that is methodological as much as it is predictive: **how to construct a decision support system for FX whose every displayed quantity is traceable to a measurement under a pre-registered evaluation protocol, which states the axis on which it has skill, which quantifies the strength of its negative findings, and which converts calibrated uncertainty into actionable risk limits.** The closed phases established what does and does not carry information; the remaining work consists of completing a small number of explicitly identified residual validation items within the existing protocol (rather than reopening its settled conclusions), performing the single sealed-set evaluation, and finalising the system and its documentation.

This is deliberately a systems thesis as much as a statistics thesis: the pre-registered protocol above is only a decision-support *system* if every one of its numbers is produced, versioned, and served by software that operates continuously rather than by a script run once for a report. The data pipeline, computation layer, API, and web application described in the Proposed Method section below are therefore treated as a primary deliverable, developed and extended across the full sixteen-week timeline rather than assembled at its end.

## Market Survey

Before specifying the proposed system, the group surveyed the retail FX decision-support tools that already compete for the same user attention this thesis targets, so that the contribution claimed below is stated relative to what exists rather than in the abstract.

- **Signal marketplaces on trading platforms** (e.g., MQL5/MetaTrader signal and expert-advisor marketplaces, TradingView public scripts) publish a directional call and a track record computed on the vendor's own in-sample selection, with no pre-registered protocol, no multiple-testing correction across the many scripts a user can browse, and no reported minimum detectable effect size for a "no edge" verdict.
- **Portfolio-analytics services** (e.g., Myfxbook, FX Blue) report realized statistics of a user's own trade history but do not issue forward-looking calibrated probabilities or a stop-out probability by holding horizon; risk figures, where present, are descriptive rather than backtested against formal coverage tests.
- **Broker-supplied "confidence" or "sentiment" widgets** attach a qualitative label (e.g., a percentage of retail positions long/short) to a pair without stating a proper scoring rule, a baseline, or a calibration diagnostic, so the number cannot be interpreted as a probability.

None of the surveyed categories discloses (a) the hypothesis space searched, (b) the multiple-testing correction applied, (c) a calibration or coverage guarantee for its stated confidence, or (d) a quantified probability of ruin behind its position-size suggestion — properties long identified as missing from applied decision support systems generally [27] and which this thesis is designed to provide for the FX case specifically. This gap, rather than a claim of superior raw directional accuracy, is the specific contribution the proposed system targets.

## Research Objectives

The primary objective of this research is to design, implement, and empirically validate a decision support system for foreign exchange trading in which forecast skill is established per target axis under strict multiple-testing control, uncertainty is communicated through calibrated conformal prediction sets, and position sizing is constrained by an estimated probability of ruin under explicitly stated assumptions.

Consistent with this objective, three groups of quantities are kept conceptually distinct throughout the study: **predictive axes** (direction, magnitude, and tail-event probability, each scored against its own baseline and its own proper scoring rule), **calibration diagnostics** (coverage and reliability of the probabilistic and conformal outputs), and the **decision layer** (Value-at-Risk, Expected Shortfall, stop-out probability, and position sizing, which consume the predictive and calibration outputs but are not themselves treated as an additional predictive claim).

Specific objectives include:

- **Establishing axis-specific predictability.** Determine, under a single fixed evaluation protocol, on which of the three target axes — direction, magnitude, and risk — out-of-sample predictive skill can be demonstrated for daily FX, and report the result on each axis separately rather than through a single aggregate score.

- **Benchmarking model complexity against a well-specified econometric baseline.** Compare linear HAR-type specifications with tree-based ensembles, recurrent and attention-based neural architectures, tabular and time-series foundation models, and systematic forecast combinations, using proper scoring rules and formal model-comparison tests.

- **Quantifying the incremental value of additional information layers.** Evaluate whether textual central-bank communication, macroeconomic and cross-asset exogenous variables, and daily candlestick geometry add measurable predictive value beyond price history, and determine whether *causal/temporal filtering* of such variables outperforms ordinary validation-based feature selection.

- **Delivering calibrated uncertainty.** Implement and validate adaptive conformal prediction so that the prediction sets displayed to the user are designed to maintain target coverage under distribution shift, verify this with empirical and regime-conditional coverage under the assumptions of the selected conformal procedure, and assess calibration with proper scoring rules and reliability diagnostics.

- **Converting forecasts into risk-controlled decisions.** Produce a per-session decision record comprising forecast volatility, recommended stop distance, probability of stop-out by holding horizon, Value-at-Risk and Expected Shortfall with formal backtests, and a ruin-constrained position size adjusted for portfolio correlation and measured slippage.

- **Reporting negative findings with quantified power.** For every negative result, report the minimum detectable effect size of the corresponding testing procedure, so that each negative claim is bounded rather than open-ended.

- **Engineering a production decision support architecture.** Design and operate, throughout the sixteen-week timeline rather than only at its conclusion, a five-layer software architecture — data pipeline, computation layer, versioned forecast artefacts, a documented API, and a web application — so that each research layer above is integrated as soon as it is completed instead of being connected for the first time late in the project (see Proposed Method and System Architecture).

- **Implementing the complete system.** Deliver a reproducible web-based decision support system in which every displayed number is traceable to a specific measurement and its supporting evidence, extending the API and web application that are already operating on a daily schedule (Introduction).

## Research Scope

This research focuses on **daily-horizon decision support for six major currency pairs involving the U.S. dollar**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, and USD/CHF (the dollar is the quote currency in the first three pairs and the base currency in the last three). The development sample covers 2010–2025 and is partitioned temporally into a training segment, a validation segment used only for model and hyper-parameter selection, and a held-out development-test segment that is scored once and is distinct from the sealed evaluation set described below.

A **sealed evaluation set** — six cross pairs not used anywhere in development, namely EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY, EUR/CHF, and NZD/USD, together with 2026 data as available at the pre-specified cutoff date for all twelve pairs — is reserved and is opened exactly once, after the final configuration has been frozen in writing; because the sealed set includes calendar year 2026 while the study concludes before that year ends, "2026" refers to whatever portion of the year has been collected through the continuously operating daily pipeline by the cutoff date, not the completed calendar year. No retuning is permitted after the sealed set is opened.

The primary data source is aggregated one-minute historical bar data, from which daily bars and five-minute realized measures (realized variance, realized quarticity, bipower variation, and signed semivariances) are computed. Supplementary sources include an official central-bank meeting and macroeconomic release calendar, daily exogenous series obtained from a public economic data service, published monetary-policy surprise measures, interest-rate differentials, and transaction-cost measurements derived from tick data with bid–ask quotations.

Forecast horizons considered are one, five, and twenty trading sessions. The one-session horizon is the operational default; longer horizons are reported with their measured skill level rather than presented as equivalent.

The study is explicitly **not** an attempt to construct a profitable automated trading strategy. Order execution, order-book microstructure modelling, market impact of large orders, intraday trade timing below the daily decision, and portfolio optimisation across asset classes other than FX are outside the scope. Statements about causality are limited to temporal predictive relationships in the Granger sense and are not presented as structural causal claims.

## Research Subjects

The primary research subject is **the statistical validation of decision support outputs for foreign exchange trading under a low signal-to-noise ratio**.

The study examines how forecast skill can be attributed to a specific target axis and to a specific information layer; how multiple-testing control, power analysis, leakage protection, and sealed-set validation interact to produce claims that survive replication; how calibrated probabilistic forecasts and conformal prediction sets can be converted into risk limits with a measured probability of ruin; and how such evidence should be presented in a decision support interface so that the user can distinguish what the system measures from what it merely assumes.

## Proposed Method and System Architecture

This section states explicitly what is proposed, distinguishing components already implemented and frozen during the pre-registered phases described in the Introduction from components that remain to be completed under this thesis. Every item below is labelled **(implemented)** or **(proposed)** accordingly; nothing in this section is presented as a finished result until it appears with its measured numbers in the Expected Results section or, later, the defended thesis.

### Proposed Predictive and Risk Layers

- **Magnitude model — extended HAR.** *(implemented, frozen as production baseline)* A heterogeneous-autoregressive specification on realized variance [7], extended with measurement-error correction [8] and jump/semivariance decomposition [9], selected over fourteen ML/DL challengers and 509 forecast combinations under the Model Confidence Set [12].
- **Direction/probability model — online expert combination.** *(implemented, frozen as production baseline)* A Hedge-style online combination [13] of four sub-models (climatology, persistence, σ̂-only, σ̂+regime), re-weighted after every session; challenged by regularised linear, tree, recurrent, attention, and foundation-model alternatives under an identical information set.
- **Causal discovery module.** *(implemented — validation role; forecast-integration role tested and deferred)* Five independently implemented methods triangulate which exogenous variable is trusted; using the estimated causal effect to directly adjust the production forecast was tested and found not yet effective. Full status in Causal Discovery Method below and Deferred / Future Work.
- **Calibration module — Mondrian-stratified adaptive conformal inference.** *(implemented for the production three-class output)* ACI [19] stratified by volatility regime [20], measured at a 1.2-percentage-point maximum calibration deviation against 2.4–3.2 points for four alternative methods. *(proposed extension)* A PID-controller-based online conformal update [30] is evaluated on the development-test segment as a candidate refinement for the below-target coverage observed while an account is drawing down (Research Methodology, Calibration, Risk, and Decision Layer).
- **Risk/decision module.** *(implemented)* Empirical-quantile VaR/ES with Kupiec [21], Christoffersen [22], and dynamic-quantile [23] backtests and a strictly consistent joint scoring function [24]; position size as a Kelly-derived growth-optimal fraction [25] capped by an estimated probability of ruin. *(proposed remaining work)* Portfolio-correlation adjustment and the slippage coefficient measured from observed stop-loss executions.

### Causal Discovery Method — Triangulated Validation, Deferred Forecast Integration

*(Status note: this subsection reports work that has already been carried out, not a plan — an earlier draft of this document proposed it as future work; five methods have since been run and the findings below are what they showed.)*

The exogenous-variable layer requires a method that (a) searches many candidate lagged predictors simultaneously rather than one pair at a time, (b) distinguishes a direct temporal link from a link that only appears because two series share a common driver or an indirect path through a third variable, and (c) plugs into the same false-discovery-control machinery already used for the price-pattern families. The **official decision gate** for the frozen exogenous feature-selection ablation (Introduction; Expected Results) is Granger-type nested-regression F-testing under step-down Westfall–Young control [4] — the same family-wise-error machinery as every other discovery family in this thesis.

**Answering "which algorithm, and how do you know it is real causality rather than random correlation" — five independently implemented methods, not one.** Beyond the official Granger gate, four additional methods from their original authors' own libraries (not self-written) were run on the same frozen data to triangulate the result:

1. **PCMCI** [18], run with the `tigramite` package (Jakob Runge's own implementation), conditioning each candidate simultaneously on the others rather than one at a time. This is the concrete mechanism that separates a genuine driver from a confound: tested alone, gold-price implied volatility (GVZCLS) is a significant predictor of FX volatility; conditioned jointly on the VIX, that significance disappears — both are proxies for the same global risk-aversion factor, and only PCMCI's simultaneous conditioning exposes this, which pairwise Granger testing structurally cannot. VIX itself survives simultaneous conditioning across every pair tested.
2. **Double Machine Learning** [33] (`doubleml`, gradient-boosted nuisance functions, five-fold cross-fitting), used to estimate the *size* of VIX's effect on next-session log realized variance rather than merely its existence. The effect is positive and Holm-significant in **6 of 6** development pairs (θ from 0.070 to 0.186, every 95% confidence interval excluding zero) — the strongest and most consistent result in the triangulation, clearing the ≥5/6-pair stability bar applied everywhere else in this thesis.
3. **Causal Forest** [34] (`econml`), used to test whether VIX's effect is regime-dependent. The effect is heterogeneous (confidence intervals differ materially by volatility regime), but the pattern is not consistent across pairs — 4 of 6 pairs show the effect increasing with stress, 2 do not — so no single "turn on the overlay under stress" rule generalises across all six pairs.
4. **CausalImpact** [28] (`pycausalimpact`, the original Bayesian structural time-series implementation), applied at the level of individual ECB meetings rather than pooled across the series — a different question ("did *this* meeting push volatility past its counterfactual") from the other three methods' pooled feature-level question. Piloted on the twenty most recent ECB meetings inside the training/validation segment: 6 of 20 (30%) show a significant post-meeting volatility increase after FDR correction, against a ~5% false-positive rate expected under pure noise, and all six are positive — evidence, not noise, though the covariate series used does not fully remove shared global-risk-aversion confounding, a limitation the method's own diagnostics flag.

Four independent methods, built by different authors on different statistical assumptions (linear/nonlinear, pairwise/simultaneous, existence/magnitude/heterogeneity/single-event), converging on the same variable is stronger evidence than any one method's p-value — this triangulation, not a single algorithm, is how this thesis answers the causality-versus-correlation question. Consistent with the Research Scope, none of the five methods is presented as establishing structural causality; every statement remains a temporal-precedence (Granger-type) or potential-outcomes claim conditioned on the assumptions each method states.

**The result splits into two roles, and only one is ready to ship.**

- **Validation role — implemented, already load-bearing.** The triangulation above is the actual mechanism that decided which exogenous variable survives into the frozen feature set: VIX is kept and ten other candidates are not, specifically because VIX — and only VIX — survives simultaneous conditioning in PCMCI and clears the 6/6-pair bar in Double ML. This is not a supplementary illustration; it is the filter, and it is done.
- **Forecast-performance role — tested, found not yet effective, deferred rather than shipped.** Using the estimated causal effect to directly adjust the production volatility forecast — a VIX-conditioned multiplicative overlay on the HAR baseline, tested both as a constant adjustment and as a Causal-Forest regime-conditional adjustment — produced **no statistically significant QLIKE improvement in any of the six pairs after Holm correction**, despite the underlying effect being real and robust. The most defensible reading is that the HAR baseline already absorbs most of this information indirectly (realized volatility over the past day/week/month is already elevated when the VIX is elevated), leaving little additional forecast value for VIX to add on top. Consistent with this thesis's commitment to reporting negative findings rather than forcing a claimed improvement that is not there, **this application is not deployed and is moved to Deferred / Future Work below rather than scheduled for the remaining timeline.**

### Proposed System Architecture

*(Data pipeline and API: implemented and operating on a daily schedule; extensions listed in the Research Plan are proposed.)*

The system is organised as five layers sharing one computation layer across two deployment targets, so the numbers a user sees on the live site and the numbers computed for the thesis are produced by the same code path:

1. **Data sources → data pipeline (`collect/`).** A one-time historical bootstrap (`histdata_dl`, `prep_fx`, `rv5`, `rv_advanced`) plus a scheduled daily job (`live_fx.py`, `lich_su_kien.py`, run four times a day under continuous-integration scheduling) construct daily bars and five-minute realized measures for the six development pairs, plus the macro/central-bank calendar.
2. **Data store (`data/`).** Static historical CSVs for the frozen development sample are kept separate from a daily-overwritten "live" store, so the sealed-evaluation and development segments cannot be silently mixed by a pipeline bug.
3. **Computation layer (`src/` and `api/cache.py`).** Chains the series-join, the extended-HAR forecast, the three-class combination, conformal calibration, and the risk/position-sizing functions in one Python process, with no serialisation boundary between the statistical code and the layer that serves it — removing an entire class of train/serve skew.
4. **API layer (`api/`, FastAPI).** One router per concern (`meta`, `market`, `forecast`, `risk`, `admin`) with Pydantic response models, already exposing the endpoints extended throughout the Research Plan below (`/health`, `/meta`, `/series`, `/indicators`, `/forecast`, `/forecast_series`, `/forecast_next`, `/journal`, `/calibration`, `/models`, `/risk`, `/events`, `/refresh`).
5. **Web application.** A single HTML/JS template (`web/ui_template.html`) is the one source of truth for the interface and is compiled to three deployed variants rather than maintained as three separate hand-written pages.

**The two deployment targets are a stated engineering trade-off, not an inconsistency — this is the answer to how the modules are linked at the deployment level.** The local/continuous-integration deployment runs the full FastAPI process (with `scipy`/`pandas`) and is the only place risk numbers are *computed*. The Vercel serverless deployment cannot host that dependency stack under its packaged-function size limit, so the daily job pre-computes the numbers that change once a day (forecast, three-class probabilities, VaR/ES) and ships them as static JSON, while a separate lightweight serverless function computes only the intraday candlestick data needed to draw the chart. Concentrating computation in one synchronous service that a lighter deployment target consumes as a versioned, materialised artefact follows established practice for scaling ML-serving systems under heterogeneous deployment constraints [31], [32].

**Module contract — the mechanism, not just the diagram, behind "how are the modules linked."** Each row below is a fixed data contract enforced in code, not an informal call:

| From | To | Via | Format |
|---|---|---|---|
| `collect/*.py` | `data/` | direct CSV/JSON write | fixed columns, no silent schema change |
| `data/` | `api/cache.py` | fixed-path CSV read | `Date, open, high, low, close, rv5, …` |
| `src/*.py` (forecast, calibration, risk) | `api/cache.py` | in-process function call | numpy/DataFrame, no serialisation |
| `api/cache.py` | `api/routers/*.py` | in-memory dictionary | Python dict (`pan`, `xs`, `sig`, `che_do`, …) |
| `api/routers/*.py` | web client | HTTP JSON | Pydantic `response_model` where declared |
| daily job | static site data | HTTP call to the API | identical JSON to the live API — never recomputed by a second formula |

The daily job never recomputes a number through a second code path: it calls the API and snapshots the response. This single-source-of-truth discipline — recommended practice for operationalising ML systems reliably [31] — is the concrete mechanism that keeps the static and live deployments from ever diverging, and is applied uniformly to every module added under the Research Plan below, not only to the modules that exist today.

## Research Methodology

The research is conducted through the following main activities.

### Literature Review

- Review realized-volatility modelling, covering the HAR family and its measurement-error, jump, and semivariance extensions, and recent comparative evidence on machine learning and foundation models for volatility forecasting.
- Review the evidence on FX directional predictability, including large-scale technical-rule studies with data-snooping control, and the momentum-factor explanation of residual rule profitability.
- Review statistically sound pattern discovery, family-wise error control, superior predictive ability testing, and model confidence sets.
- Review conformal prediction under distribution shift, proper scoring rules, and calibration diagnostics.
- Review Value-at-Risk and Expected Shortfall backtesting, and growth-optimal position sizing under a ruin constraint.

### Data Preparation and Protocol Fixing

- Construct daily bars and five-minute realized measures from one-minute source data; verify contiguity, session conventions, and time-zone handling.
- Fix the temporal partition, the evaluation metrics per axis, the multiple-testing procedure, and the sealed-set policy in a written record **before** any scoring, and record every configuration evaluated on the development sample.
- Implement a permanent anti-leakage protocol: every derived feature must satisfy the self-test that truncating all information revealed after time *t* leaves its value at *t* unchanged.

### Baseline Establishment and Model Comparison

- Establish the extended HAR specification as the production magnitude baseline and an online expert-combination layer [13] as the three-class probability baseline.
- Compare against linear regularised models, gradient-boosted trees, recurrent and attention-based networks, a tabular foundation model [14], zero-shot time-series foundation models [11], and systematic forecast combinations, all under an identical information set, refit frequency, and variance-conversion procedure.
- Evaluate with scale-invariant QLIKE [15], the continuous ranked probability score, and the logarithmic and Brier scores [16], and assess significance with Diebold–Mariano tests [17] and Model Confidence Sets [12].

### Pattern, Exogenous, and Causal Analysis

- Enumerate the hypothesis space in advance for every discovery family — symbolic sequence patterns, motif and matrix-profile analysis, interpretable rule learning, regime models, textual central-bank features, exogenous macro and cross-asset variables, and daily candlestick geometry — and apply step-down max-*T* resampling with block-preserving null models [4]. Remaining work in this family is limited to completing pre-registered evaluations at horizons and axes not yet scored (Section on the Research Plan); no new discovery family will be introduced after the evaluation protocol has been frozen.
- Condition every candidate on the strongest available null: the production volatility forecast for the magnitude axis, and time-series momentum for the direction axis [3].
- For the exogenous layer, contrast three feature-selection regimes under an identical model class and parameter budget: all declared variables, validation-selected variables, and variables filtered by Granger-type causal testing under step-down Westfall–Young control [4], triangulated against four independently implemented methods — PCMCI [18], Double Machine Learning [33], Causal Forest [34], and CausalImpact [28] — with a cross-pair and cross-year stability screen (full triangulation and its two-role finding detailed in Proposed Method and System Architecture: Causal Discovery Method), in order to isolate the contribution of causal filtering itself. This layer, including the triangulation, is complete; remaining work is limited to writing up the findings (Research Plan, Week 9–10) — no new causal discovery work is scheduled, and using the causal effect to adjust the forecast directly is deferred (Deferred / Future Work) rather than pursued in the remaining timeline.
- Screen surviving candidates for sign consistency across pairs and across years, and require leave-one-pair-out transfer before any candidate is admitted.

### Calibration, Risk, and Decision Layer

- Implement adaptive conformal inference with Mondrian stratification by volatility regime [19], [20], and verify empirical coverage overall and conditional on regime.
- Estimate Value-at-Risk and Expected Shortfall from empirical quantiles of standardised returns scaled by the volatility forecast, and backtest per pair using unconditional coverage [21], independence and conditional coverage [22], and dynamic quantile tests [23]; assess the joint (VaR, ES) pair with a strictly consistent scoring function [24].
- Compute position size as the minimum of a growth-optimal fraction [25] and a cap constrained by an estimated probability of ruin, adjusted by volatility regime, drawdown state, a portfolio correlation factor, and a slippage coefficient measured from observed stop-loss executions; the ruin estimate is reported together with the return, dependence, and sizing assumptions on which it is conditioned.

### Power Analysis and Sealed Validation

- For every negative finding, inject synthetic effects of known magnitude through the actual testing funnel and report the effect size detected with 80% power, together with a negative control establishing the false-positive rate under pure noise.
- Freeze the final configuration in writing, then open the sealed evaluation set exactly once and report the resulting numbers without modification.

### System Implementation and Analysis

- Maintain, throughout the study rather than only at its conclusion, the operating end-to-end product path specified in Proposed Method and System Architecture — data ingestion, production forecast artefact, application programming interface, and web application — so that each research layer is integrated as a versioned artefact update the same week it is completed (see the per-week engineering deliverables in the Research Plan) instead of being connected for the first time late in the project; dedicate the final weeks to hardening this system with a computation layer, a versioned forecast-artefact interface, and a user interface that presents per-axis skill, conformal prediction sets, the event calendar with measured historical responses, and the risk decision record with full provenance.
- Analyse failure cases and limitations, including pairs that fail tail backtests and horizons at which calibration degrades, and report them within the interface rather than omitting them.

## Expected Results and Contributions

The expected outcomes of this research include:

- **A validated decision support system for FX** that issues, for each session and each pair, a volatility forecast, a calibrated three-class probability distribution with a conformal prediction set, an event-risk assessment, and a ruin-constrained position-sizing recommendation, with every displayed quantity traceable to a specific measurement.

- **An axis-resolved characterisation of daily FX predictability**, establishing where skill exists and where it does not, under a single protocol applied uniformly across all information layers. Results obtained and frozen to date indicate positive and statistically significant skill on the magnitude axis and no detectable skill on the direction axis across twelve independent discovery branches, including four independent representations of price history and of official central-bank communication; the thesis will complete this analysis at the remaining horizons and confirm it on the sealed evaluation set, which has not yet been opened.

- **A systematic benchmark of model complexity for FX volatility forecasting**, covering linear, tree-based, recurrent, attention-based, tabular-foundation and time-series-foundation models, together with exhaustive forecast combinations, evaluated with proper scoring rules and model confidence sets.

- **An empirical comparison of causal filtering against ordinary feature selection.** A completed, frozen ablation — its sealed-set evaluation deliberately not opened, per the pre-registered rule that a non-positive development-test verdict does not warrant it (Introduction) — shows that variables filtered by temporal-causal testing with a stability screen degrade substantially less out of sample than variables selected by validation performance, with the mechanism identified and measured as distribution drift in the selected macroeconomic variables. This comparison appears to be underexplored in the FX forecasting literature.

- **A five-method causal triangulation, and an honest split between its two roles.** Granger+Westfall–Young, PCMCI, Double Machine Learning, Causal Forest, and CausalImpact independently converge on the same variable (VIX) and concretely demonstrate the mechanism — simultaneous conditioning in PCMCI — that separates it from a confound (gold-implied volatility) that looks causal only until conditioned on VIX. The validation role of this triangulation (deciding which exogenous variable is trusted) is complete and already load-bearing in the frozen feature set; its forecast-performance role (directly adjusting the production forecast) was tested and found not yet statistically effective in any of the six development pairs, and is reported as a negative finding rather than folded into a claimed improvement (Proposed Method and System Architecture; Deferred / Future Work).

- **Power-bounded negative evidence.** Each negative finding will be reported together with the minimum effect size detectable by the corresponding procedure, converting statements of the form "no relationship was found" into bounded claims of the form "no relationship stronger than *X* exists in these data".

- **Three applied contributions to risk-aware decision support**: a portfolio correlation factor for ruin-constrained sizing, quantifying the gap between the nominal and the realised probability of ruin when multiple correlated positions are opened simultaneously; a holding-horizon table on the decision record, addressing the systematic misreading of single-session stop-out probabilities; and a slippage coefficient estimated from observed stop-loss executions rather than assumed.

- **A reproducible research record** containing the full data dictionary, the sealed-set protocol, the registry of every model configuration and hypothesis evaluated, and reproduction commands for every reported number, including all negative results.

- **A documented, versioned software architecture** — the five-layer data pipeline, computation layer, API, and web application specified in Proposed Method and System Architecture — with an explicit module contract between every layer, so that the engineering contribution (how the numbers are produced, versioned, and served without the static and live deployments diverging) is itself reported and defensible, not only the numbers it serves.

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
| 1 | – Consolidate the data pipeline and verify contiguity, session conventions and time-zone handling. <br> – Fix the evaluation protocol: temporal partition, per-axis metrics, multiple-testing procedure, and sealed-set policy. <br> – Reproduce all baseline results from a clean checkout. <br> – **Engineering:** verify the already-operating daily pipeline job and its `/health`/`/meta` diagnostics against the reproduced baseline; tag this state as the architecture's starting version. | Week 1–2 |
| 2 | – Complete the pattern-discovery families remaining at the five- and twenty-session horizons. <br> – Apply superior predictive ability testing at the family level across all discovery branches. <br> – Document the enumerated hypothesis space in full. <br> – **Engineering:** register the newly completed discovery-family results in the model registry served by `/models`. | Week 3–4 |
| 3 | – Conduct minimum detectable effect size analysis for every negative finding. <br> – Validate the discovery funnel with negative controls and known-effect injection. <br> – Formalise the reporting template for power-bounded negative claims. <br> – **Engineering:** extend `/calibration` with the minimum-detectable-effect-size figure for every negative finding, so the power-bounded claim is traceable through the same API as every other number. | Week 5–6 |
| 4 | – Address tail-risk failures for the two pairs that do not pass Value-at-Risk and Expected Shortfall backtests at the 99% level, using conditional and extreme-value approaches. <br> – Recalibrate the twenty-session horizon and re-measure calibration error. <br> – **Engineering:** update the `/risk` endpoint's VaR/ES backtest fields for the two affected pairs and the `/forecast_next` twenty-session horizon. | Week 7–8 |
| 5 | – Write up the completed exogenous/causal analysis for the thesis: the frozen all-variable/validation-selected/causally-filtered ablation and the five-method causal triangulation (Proposed Method, Causal Discovery Method) — this work is done, not scheduled to be run here. <br> – Report results separately on the direction, magnitude and risk axes, as already measured. <br> – No new causal-discovery hypotheses are introduced at this stage, consistent with the frozen evaluation protocol; the causal-conditioned forecast overlay tested and found not yet effective stays in Deferred / Future Work rather than being reopened. <br> – **Engineering:** expose the causal-validation provenance (which exogenous variable passed or failed the confounding check, and why) as a read-only diagnostic field in `/forecast`'s provenance metadata — explicitly not used to alter the forecast value, consistent with the causal layer's validation-only role today. | Week 9–10 |
| 6 | – Evaluate the end-to-end economic result after measured transaction costs and slippage. <br> – Accumulate the forecast journal and compute rolling calibration on issued forecasts. <br> – **Engineering:** extend `/journal` with realised transaction-cost-adjusted performance and rolling calibration diagnostics. | Week 11 |
| 7 | – Freeze the final configuration in writing, including variables, selection method, models, metrics and decision rules. <br> – Prepare the sealed-set evaluation record. <br> – **Engineering:** tag and freeze the API response schemas and forecast-artefact version alongside the written protocol freeze. | Week 12 |
| 8 | – Open the sealed evaluation set exactly once and score the frozen configuration. <br> – Report results without modification and without retuning. <br> – **Engineering:** serve the sealed-set results through the same frozen `/forecast`, `/risk`, and `/calibration` endpoints used throughout — no separate reporting path. | Week 13 |
| 9 | – Complete the system: computation layer, versioned forecast artefacts, application programming interface, and user interface with full provenance for every displayed quantity. <br> – Verify continuous daily operation. | Week 14 |
| 10 | – Analyse model behaviour by volatility regime, by pair and by year. <br> – Examine representative failure cases and document limitations. <br> – **Engineering:** add regime-, pair-, and year-conditional breakdowns to the web application's risk and calibration views. | Week 15 |
| 11 | – Complete the thesis report. <br> – Revise content based on advisor feedback. <br> – Prepare for the final defence. | Week 16 |

Software-architecture work therefore runs across Weeks 1–15, extending the already-operating API and web application (Proposed Method and System Architecture) one layer at a time as each research result is frozen, rather than being concentrated in a single week at the end.

## Deferred / Future Work (Not Scheduled in This Thesis)

Some directions surfaced by completed experiments are explicitly **not** part of the sixteen-week Research Plan above. Listing them here, rather than folding them into the Proposed Method section in a way that would imply they are being built during this thesis, keeps the timeline honest about what is actually committed:

- **Causal-conditioned forecast overlay.** The VIX→volatility causal effect is real and robust (Double Machine Learning, 6/6 pairs significant after Holm correction; Proposed Method, Causal Discovery Method), but using it to directly adjust the production HAR forecast — tested both as a constant multiplicative overlay and as a Causal-Forest regime-conditional overlay — produced no statistically significant QLIKE improvement in any pair. Deferred pending either a better functional form for combining the causal signal with the HAR forecast, or acceptance that the baseline already absorbs it and no overlay is warranted.
- **Contemporaneous causal edges (PCMCI+, τ_min = 0).** An exploratory run found several undirected or sign-ambiguous contemporaneous edges between implied-volatility variables and FX volatility, inconsistent in structure with the frozen lagged (τ ≥ 1) result. Resolving the directionality needs a larger sample or a nonlinear conditional-independence test; not attempted in this thesis.
- **Cross-pair volatility spillover (DYNOTEARS).** A self-implemented DYNOTEARS network (the `causalnex` reference implementation does not support the project's Python version) detects contemporaneous spillover structure between pairs (e.g., AUD/USD↔USD/CAD, EUR/USD→USD/CHF), consistent with an independently run linear Diebold–Yilmaz spillover measure. Neither the linear benchmark nor the DYNOTEARS-derived edges, added as an extra HAR feature, improved test-set QLIKE (six-pair average 0.1281 without vs. 0.1285 with the added column). The structure appears real but is not yet forecastable at a daily horizon with the methods tried; revisiting with a different estimation window or a nonlinear specification is future work.
- **Revisiting the exogenous/causal layer if FX-specific implied volatility becomes available.** The single most promising untested candidate, `EVZCLS` (EUR/USD-specific implied volatility), was discontinued by CBOE in March 2025 and is unavailable for the sample used here.

None of the above blocks the sealed-set evaluation or the system hardening in Weeks 12–15: the production magnitude forecast for this thesis continues to rely on the extended-HAR baseline alone, with the causal layer contributing to feature-trust validation (implemented) but not to the forecast value itself (deferred).

## Work Assignment

*(DRAFT SPLIT — TO BE CONFIRMED BY THE TWO STUDENTS BEFORE SUBMISSION. The advisor explicitly rejected an undifferentiated 50/50 split; the allocation below is a working draft aligned with the codebase's own module boundaries — Member A on the quantitative/statistical core, Member B on the systems/software side — and must be replaced with the two members' actual division of labour, not merely re-approved as written.)*

| Assignments | Member A (Nguyen Quoc Huy) | Member B (Huynh Tran Quoc Huy) |
|---|---|---|
| Review literature and define research scope | Lead | Support |
| Construct the data pipeline and realized measures | Support | Lead |
| Fix the evaluation protocol and sealed-set policy | Joint | Joint |
| Develop and benchmark the volatility forecasting layer | Lead | Support |
| Develop the pattern-discovery and symbolic-representation branch | Lead | Support |
| Implement multiple-testing control and stability screening | Lead | Support |
| Conduct minimum detectable effect size and power analysis | Lead | Support |
| Develop the exogenous, causal (five-method triangulation), and cross-asset information layers | Lead | Support |
| Implement conformal prediction and calibration diagnostics | Support | Lead |
| Develop the risk layer: VaR/ES backtesting and position sizing | Lead | Support |
| Conduct the end-to-end economic evaluation | Joint | Joint |
| Execute the sealed-set validation | Joint | Joint |
| Design and implement the data pipeline scheduling and module contracts | Support | Lead |
| Implement the API layer (FastAPI routers, response schemas) | Support | Lead |
| Implement the web application and its three deployment variants | Support | Lead |
| Analyse failure cases and document limitations | Joint | Joint |
| Write and revise the thesis report | Joint | Joint |
| Prepare for the final defence | Joint | Joint |

"Lead" denotes primary ownership and accountability for a task; "Support" denotes review, testing, and secondary contribution; "Joint" denotes tasks genuinely carried out together (protocol decisions, evaluation, writing, and defence) where a lead/support split would misrepresent how the work is actually done. Both members remain jointly responsible for the correctness of the full system regardless of individual lead assignments.

<!-- KHOI-KY -->
<!-- Bo sinh .docx (src/xuat_proposal.py) tu dung o ky hai cot tu day.
     Khong viet bang &nbsp; nua — no khong len duoc .docx cho dep. -->
