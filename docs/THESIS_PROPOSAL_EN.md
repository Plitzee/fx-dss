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

Preliminary work carried out by the group has established the feasibility and the empirical baseline for this design. A complete data pipeline has been constructed for six major currency pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF) over 2010–2025, aggregating approximately 34.9 million one-minute bars into daily bars and five-minute realized measures; a production volatility layer, a three-class probability layer, and a risk layer have been implemented end to end. Under a fixed 70/15/15 temporal split, the extended HAR specification attains a test-set QLIKE of 0.1585 against 0.2172 for the previous baseline, while fourteen machine-learning and deep-learning alternatives, two zero-shot time-series foundation models, and 509 forecast combinations fail to displace it from the Model Confidence Set [12]. On the direction axis, 8,652 fully enumerated rule hypotheses across twelve independent discovery branches yield zero survivors after family-wise error control, with a measured detection power corresponding to a lift of 1.20.

This thesis therefore addresses a research problem that is methodological as much as it is predictive: **how to construct a decision support system for FX whose every displayed quantity is traceable to a measurement under a pre-registered evaluation protocol, which states the axis on which it has skill, which quantifies the strength of its negative findings, and which converts calibrated uncertainty into actionable risk limits.** The remaining work consists of completing the outstanding validation components, extending the causal and exogenous-information analysis, performing a single sealed-set evaluation, and finalising the system and its documentation.

## Research Objectives

The primary objective of this research is to design, implement, and empirically validate a decision support system for foreign exchange trading in which forecast skill is established per target axis under strict multiple-testing control, uncertainty is communicated with finite-sample coverage guarantees, and position sizing is constrained by a measured probability of ruin.

Specific objectives include:

- **Establishing axis-specific predictability.** Determine, under a single fixed evaluation protocol, on which of the three target axes — direction, magnitude, and risk — out-of-sample predictive skill can be demonstrated for daily FX, and report the result on each axis separately rather than through a single aggregate score.

- **Benchmarking model complexity against a well-specified econometric baseline.** Compare linear HAR-type specifications with tree-based ensembles, recurrent and attention-based neural architectures, tabular and time-series foundation models, and systematic forecast combinations, using proper scoring rules and formal model-comparison tests.

- **Quantifying the incremental value of additional information layers.** Evaluate whether textual central-bank communication, macroeconomic and cross-asset exogenous variables, and daily candlestick geometry add measurable predictive value beyond price history, and determine whether *causal/temporal filtering* of such variables outperforms ordinary validation-based feature selection.

- **Delivering calibrated and guaranteed uncertainty.** Implement and validate adaptive conformal prediction so that the prediction sets displayed to the user carry a finite-sample coverage guarantee under distribution shift, and verify calibration with proper scoring rules and reliability diagnostics.

- **Converting forecasts into risk-controlled decisions.** Produce a per-session decision record comprising forecast volatility, recommended stop distance, probability of stop-out by holding horizon, Value-at-Risk and Expected Shortfall with formal backtests, and a ruin-constrained position size adjusted for portfolio correlation and measured slippage.

- **Reporting negative findings with quantified power.** For every negative result, report the minimum detectable effect size of the corresponding testing procedure, so that each negative claim is bounded rather than open-ended.

- **Implementing the complete system.** Deliver a reproducible web-based decision support system in which every displayed number is traceable to a specific measurement and its supporting evidence.

## Research Scope

This research focuses on **daily-horizon decision support for six major USD-quoted currency pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, and USD/CHF. The development sample covers 2010–2025 and is partitioned temporally into training, validation, and test segments in a 70/15/15 ratio; model and hyper-parameter selection uses only the validation segment, and the test segment is scored once.

A **sealed evaluation set** — six cross pairs, NZD/USD, and the whole of 2026 — is reserved and is opened exactly once, at the end of the study, after the final configuration has been frozen in writing. No retuning is permitted after the sealed set is opened.

The primary data source is aggregated one-minute historical bar data, from which daily bars and five-minute realized measures (realized variance, realized quarticity, bipower variation, and signed semivariances) are computed. Supplementary sources include an official central-bank meeting and macroeconomic release calendar, daily exogenous series obtained from a public economic data service, published monetary-policy surprise measures, interest-rate differentials, and transaction-cost measurements derived from tick data with bid–ask quotations.

Forecast horizons considered are one, five, and twenty trading sessions. The one-session horizon is the operational default; longer horizons are reported with their measured skill level rather than presented as equivalent.

The study is explicitly **not** an attempt to construct a profitable automated trading strategy. Order execution, order-book microstructure modelling, market impact of large orders, intraday trade timing below the daily decision, and portfolio optimisation across asset classes other than FX are outside the scope. Statements about causality are limited to temporal predictive relationships in the Granger sense and are not presented as structural causal claims.

## Research Subjects

The primary research subject is **the statistical validation of decision support outputs for foreign exchange trading under a low signal-to-noise ratio**.

The study examines how forecast skill can be attributed to a specific target axis and to a specific information layer; how multiple-testing control, power analysis, leakage protection, and sealed-set validation interact to produce claims that survive replication; how calibrated probabilistic forecasts and conformal prediction sets can be converted into risk limits with a measured probability of ruin; and how such evidence should be presented in a decision support interface so that the user can distinguish what the system measures from what it merely assumes.

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

- Enumerate the hypothesis space in advance for every discovery family — symbolic sequence patterns, motif and matrix-profile analysis, interpretable rule learning, regime models, textual central-bank features, exogenous macro and cross-asset variables, and daily candlestick geometry — and apply step-down max-*T* resampling with block-preserving null models [4].
- Condition every candidate on the strongest available null: the production volatility forecast for the magnitude axis, and time-series momentum for the direction axis [3].
- For the exogenous layer, contrast three feature-selection regimes under an identical model class and parameter budget: all declared variables, validation-selected variables, and variables filtered by temporal-causal testing [18] with a cross-pair and cross-year stability screen, in order to isolate the contribution of causal filtering itself.
- Screen surviving candidates for sign consistency across pairs and across years, and require leave-one-pair-out transfer before any candidate is admitted.

### Calibration, Risk, and Decision Layer

- Implement adaptive conformal inference with Mondrian stratification by volatility regime [19], [20], and verify empirical coverage overall and conditional on regime.
- Estimate Value-at-Risk and Expected Shortfall from empirical quantiles of standardised returns scaled by the volatility forecast, and backtest per pair using unconditional coverage [21], independence and conditional coverage [22], and dynamic quantile tests [23]; assess the joint (VaR, ES) pair with a strictly consistent scoring function [24].
- Compute position size as the minimum of a growth-optimal fraction [25] and a ruin-constrained cap, adjusted by volatility regime, drawdown state, a portfolio correlation factor, and a slippage coefficient measured from observed stop-loss executions.

### Power Analysis and Sealed Validation

- For every negative finding, inject synthetic effects of known magnitude through the actual testing funnel and report the effect size detected with 80% power, together with a negative control establishing the false-positive rate under pure noise.
- Freeze the final configuration in writing, then open the sealed evaluation set exactly once and report the resulting numbers without modification.

### System Implementation and Analysis

- Implement the system as a web application with a computation layer, a versioned forecast-artefact interface, and a user interface that presents per-axis skill, conformal prediction sets, the event calendar with measured historical responses, and the risk decision record with full provenance.
- Analyse failure cases and limitations, including pairs that fail tail backtests and horizons at which calibration degrades, and report them within the interface rather than omitting them.

## Expected Results and Contributions

The expected outcomes of this research include:

- **A validated decision support system for FX** that issues, for each session and each pair, a volatility forecast, a calibrated three-class probability distribution with a conformal prediction set, an event-risk assessment, and a ruin-constrained position-sizing recommendation, with every displayed quantity traceable to a specific measurement.

- **An axis-resolved characterisation of daily FX predictability**, establishing where skill exists and where it does not, under a single protocol applied uniformly across all information layers. Preliminary results indicate positive and statistically significant skill on the magnitude axis and no detectable skill on the direction axis across twelve independent discovery branches; the thesis will complete this analysis at the remaining horizons and validate it on sealed data.

- **A systematic benchmark of model complexity for FX volatility forecasting**, covering linear, tree-based, recurrent, attention-based, tabular-foundation and time-series-foundation models, together with exhaustive forecast combinations, evaluated with proper scoring rules and model confidence sets.

- **An empirical comparison of causal filtering against ordinary feature selection.** Preliminary results show that variables filtered by temporal-causal testing with a stability screen degrade substantially less out of sample than variables selected by validation performance, with the mechanism identified and measured as distribution drift in the selected macroeconomic variables. This result is, to our knowledge, not documented in the FX forecasting literature.

- **Power-bounded negative evidence.** Each negative finding will be reported together with the minimum effect size detectable by the corresponding procedure, converting statements of the form "no relationship was found" into bounded claims of the form "no relationship stronger than *X* exists in these data".

- **Three applied contributions to risk-aware decision support**: a portfolio correlation factor for ruin-constrained sizing, quantifying the gap between the nominal and the realised probability of ruin when multiple correlated positions are opened simultaneously; a holding-horizon table on the decision record, addressing the systematic misreading of single-session stop-out probabilities; and a slippage coefficient estimated from observed stop-loss executions rather than assumed.

- **A reproducible research record** containing the full data dictionary, the sealed-set protocol, the registry of every model configuration and hypothesis evaluated, and reproduction commands for every reported number, including all negative results.

## References

[1] R. A. Meese and K. Rogoff, "Empirical exchange rate models of the seventies: Do they fit out of sample?," *Journal of International Economics*, vol. 14, no. 1–2, pp. 3–24, 1983, doi: 10.1016/0022-1996(83)90017-X.

[2] P.-H. Hsu, M. P. Taylor, and Z. Wang, "Technical trading: Is it still beating the foreign exchange market?," *Journal of International Economics*, vol. 102, pp. 188–208, 2016, doi: 10.1016/j.jinteco.2016.03.012.

[3] M. C. Hutchinson, P. Kyziropoulos, J. O'Brien, F. O'Reilly, and T. Sharma, "Technical trading rule profitability in currencies: It's all about momentum," *Research in International Business and Finance*, vol. 61, Art. no. 101779, 2022, doi: 10.1016/j.ribaf.2022.101779.

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

## Research Plan and Timeline

Research timeline: 16 weeks.

| No. | Assignments | Timeline |
|---|---|---|
| 1 | – Consolidate the data pipeline and verify contiguity, session conventions and time-zone handling. <br> – Fix the evaluation protocol: temporal partition, per-axis metrics, multiple-testing procedure, and sealed-set policy. <br> – Reproduce all baseline results from a clean checkout. | Week 1–2 |
| 2 | – Complete the pattern-discovery families remaining at the five- and twenty-session horizons. <br> – Apply superior predictive ability testing at the family level across all discovery branches. <br> – Document the enumerated hypothesis space in full. | Week 3–4 |
| 3 | – Conduct minimum detectable effect size analysis for every negative finding. <br> – Validate the discovery funnel with negative controls and known-effect injection. <br> – Formalise the reporting template for power-bounded negative claims. | Week 5–6 |
| 4 | – Address tail-risk failures for the two pairs that do not pass Value-at-Risk and Expected Shortfall backtests at the 99% level, using conditional and extreme-value approaches. <br> – Recalibrate the twenty-session horizon and re-measure calibration error. | Week 7–8 |
| 5 | – Complete the exogenous and causal information layer: all-variable, validation-selected and causally-filtered ablation under an identical parameter budget. <br> – Report results separately on the direction, magnitude and risk axes. <br> – Screen candidates for cross-pair and cross-year stability. | Week 9–10 |
| 6 | – Evaluate the end-to-end economic result after measured transaction costs and slippage. <br> – Accumulate the forecast journal and compute rolling calibration on issued forecasts. | Week 11 |
| 7 | – Freeze the final configuration in writing, including variables, selection method, models, metrics and decision rules. <br> – Prepare the sealed-set evaluation record. | Week 12 |
| 8 | – Open the sealed evaluation set exactly once and score the frozen configuration. <br> – Report results without modification and without retuning. | Week 13 |
| 9 | – Complete the system: computation layer, versioned forecast artefacts, application programming interface, and user interface with full provenance for every displayed quantity. <br> – Verify continuous daily operation. | Week 14 |
| 10 | – Analyse model behaviour by volatility regime, by pair and by year. <br> – Examine representative failure cases and document limitations. | Week 15 |
| 11 | – Complete the thesis report. <br> – Revise content based on advisor feedback. <br> – Prepare for the final defence. | Week 16 |

## Work Assignment

| Assignments | Member A | Member B |
|---|---|---|
| Review literature and define research scope | 50% | 50% |
| Construct the data pipeline and realized measures | 50% | 50% |
| Fix the evaluation protocol and sealed-set policy | 50% | 50% |
| Develop and benchmark the volatility forecasting layer | 50% | 50% |
| Develop the pattern-discovery and symbolic-representation branch | 50% | 50% |
| Implement multiple-testing control and stability screening | 50% | 50% |
| Conduct minimum detectable effect size and power analysis | 50% | 50% |
| Develop the exogenous, textual and causal information layers | 50% | 50% |
| Implement conformal prediction and calibration diagnostics | 50% | 50% |
| Develop the risk layer: VaR/ES backtesting and position sizing | 50% | 50% |
| Conduct the end-to-end economic evaluation | 50% | 50% |
| Execute the sealed-set validation | 50% | 50% |
| Implement the application programming interface and user interface | 50% | 50% |
| Analyse failure cases and document limitations | 50% | 50% |
| Write and revise the thesis report | 50% | 50% |
| Prepare for the final defence | 50% | 50% |

Work is split evenly across all activities. Both members are jointly responsible for every stage rather than owning separate stages independently, consistent with the collaborative nature of the research.

<!-- KHOI-KY -->
<!-- Bo sinh .docx (src/xuat_proposal.py) tu dung o ky hai cot tu day.
     Khong viet bang &nbsp; nua — no khong len duoc .docx cho dep. -->
