# Final evaluation and review



The chosen model is **Gradient boosting**, selected before test scoring by validation AP 0.555499. Settings, thresholds and checkpoint hashes were locked in artifacts/frozen_selection.json and verified before loading test partitions.

The shared test contains 4,500 held-out records. Positive prevalence is 0.221111; this is also the majority dummy's AP. All 30,000 records remain in the study, with exact predictor duplicates confined to one partition.



## Metrics and operating point



The definitive comparison is artifacts/evaluation/test_metrics.csv (one row per model/threshold). AP and ROC-AUC use continuous scores; threshold metrics use score >= threshold. Zero precision when no positives are predicted is explicitly defined as zero. AP is not accuracy or trapezoidal PR area.

- Majority baseline: test AP 0.221111; ROC-AUC 0.500000; threshold 0.500000; precision 0.000000; recall 0.000000; F1 0.000000. TN/FP/FN/TP = 3505/0/995/0.

- Logistic regression: test AP 0.554551; ROC-AUC 0.788147; threshold 0.250129; precision 0.517241; recall 0.603015; F1 0.556845. TN/FP/FN/TP = 2945/560/395/600.

- Gradient boosting: test AP 0.566227; ROC-AUC 0.796417; threshold 0.339053; precision 0.574376; recall 0.531658; F1 0.552192. TN/FP/FN/TP = 3113/392/466/529.

- PyTorch MLP: test AP 0.566441; ROC-AUC 0.792969; threshold 0.310109; precision 0.552605; recall 0.543719; F1 0.548126. TN/FP/FN/TP = 3067/438/454/541.



For the selected model, threshold 0.5 yields precision 0.6672, recall 0.3930, F1 0.4946, 195 false positives and 604 missed defaults. The validation-selected threshold 0.339053 yields precision 0.5744, recall 0.5317, F1 0.5522, 392 false positives and 466 missed defaults.

The review queue changes from 586 to 921 cases out of 4500. F1 was fixed as the threshold-selection objective before test evaluation. It balances precision and recall, but does not encode real review capacity or asymmetric financial costs. A lower threshold usually catches more defaults while sending more non-default cases for review. The threshold is not reoptimized on test.



## Case inspection



error_examples.csv contains three false positives, three false negatives, two true positives and two true negatives when each category exists. Examples are chosen at fixed score positions within each outcome group (10/50/90% for errors, 25/75% for correct cases), not to imply that one case explains the entire population. The source row index is a reproducibility key and is never an input to the model.

- FP, row 22052: score 0.3626, actual 0; flagged for review but did not default. Latest repayment status: documented 1-month payment delay; credit limit NT$ 50,000, latest bill NT$ 22,021, latest payment NT$ 1,000. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- FP, row 14788: score 0.4981, actual 0; flagged for review but did not default. Latest repayment status: documented 2-month payment delay; credit limit NT$ 30,000, latest bill NT$ 200, latest payment NT$ 0. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- FP, row 904: score 0.7423, actual 0; flagged for review but did not default. Latest repayment status: documented 2-month payment delay; credit limit NT$ 80,000, latest bill NT$ 57,364, latest payment NT$ 2,200. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- FN, row 27520: score 0.0803, actual 1; not prioritized but defaulted. Latest repayment status: code 0 (semantics not specified by UCI); credit limit NT$ 80,000, latest bill NT$ 89,088, latest payment NT$ 3,400. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- FN, row 19569: score 0.1690, actual 1; not prioritized but defaulted. Latest repayment status: code 0 (semantics not specified by UCI); credit limit NT$ 50,000, latest bill NT$ 10,950, latest payment NT$ 1,097. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- FN, row 15663: score 0.2959, actual 1; not prioritized but defaulted. Latest repayment status: documented 1-month payment delay; credit limit NT$ 10,000, latest bill NT$ 9,193, latest payment NT$ 0. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- TP, row 17201: score 0.4912, actual 1; flagged and defaulted. Latest repayment status: documented 2-month payment delay; credit limit NT$ 390,000, latest bill NT$ 393,415, latest payment NT$ 0. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- TP, row 21027: score 0.7264, actual 1; flagged and defaulted. Latest repayment status: documented 2-month payment delay; credit limit NT$ 50,000, latest bill NT$ 17,150, latest payment NT$ 0. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- TN, row 5098: score 0.0706, actual 0; not prioritized and did not default. Latest repayment status: code 0 (semantics not specified by UCI); credit limit NT$ 120,000, latest bill NT$ 19,544, latest payment NT$ 5,000. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.

- TN, row 10306: score 0.1702, actual 0; not prioritized and did not default. Latest repayment status: documented duly-paid status; credit limit NT$ 190,000, latest bill NT$ 88, latest payment NT$ 0. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.



False positives show that risky-looking histories do not guarantee a future default. False negatives show that recorded payment history cannot fully identify future events. Income changes and other unobserved circumstances cannot be inferred from these examples.



## Model interpretation



Permutation importance is calculated after model selection on the unchanged test set for description only. Each original column is shuffled five times with seed 42; the model is never refitted. The saved value is the mean decrease in AP; error bars are the standard deviation across permutations, not a confidence interval over clients.

The largest decreases are: PAY_0 (0.1986); BILL_AMT1 (0.0100); PAY_2 (0.0098); BILL_AMT3 (0.0095); LIMIT_BAL (0.0085).

This measures predictive reliance, not causality. Correlated monthly histories can substitute for one another, distributing or understating individual importance. Shuffling may create unusual feature combinations. Negative small values can reflect finite-sample/permutation variation. This interpretation does not trigger another tuning round.



## Neural learning and model comparison



The selected MLP restores epoch 17 of 25 epochs run. At that checkpoint train AP = 0.573401, validation AP = 0.552254, train BCE = 0.423166, validation BCE = 0.425676. At the last epoch the corresponding values are 0.580146, 0.551845, 0.420334, and 0.425317.

Training and validation curves are measured with dropout disabled. Improving train scores alongside stalled/worsening validation scores indicate overfitting; restoring the earlier validation-AP checkpoint limits it. Loss and AP can peak at different epochs because BCE assesses probabilistic fit while AP assesses ranking. Validation sampling variation also affects small fluctuations. A persistent low train and validation score would suggest underfitting; the saved curves should be read rather than assuming it from architecture alone.

The MLP test AP is 0.566441, compared with boosting 0.566227 and logistic regression 0.554551. In this run the MLP matches or exceeds boosting on test AP. This does not establish general neural superiority; the model used in the demo still follows the frozen validation decision.

The linear logistic baseline has a restricted additive decision function on transformed features, so it cannot freely model interactions. The majority baseline deliberately ignores all information and illustrates why accuracy alone would be misleading with class imbalance. A single split and small search do not establish a statistically significant universal ranking.



## Limitations and realistic next steps



The data describe historical Taiwanese credit-card clients in 2005. Evaluation is on held-out clients from this source population, not a later month, new country or new bank. The label is the workbook's next-month-default outcome; the UCI description does not provide a detailed operational delinquency definition. Features must be available by the end of September for this prediction boundary.

A future-period study needs multiple timestamped cohorts, leakage-safe feature cutoffs, consistent outcome windows and a forward temporal test. A bank-transfer study needs an external bank's compatible schema and independently labeled holdout. Both require drift and calibration assessment, subgroup performance/fairness checks, and operational review-capacity/cost analysis. Undefined codes and demographic fields deserve extra care. We do not claim calibrated individual risk, causality, fairness, or readiness for automated lending decisions.



## Verification



Frozen fingerprints verified before test loading. Every saved model was reloaded for test inference. Scores are checked finite and within [0,1]. Confusion counts sum to test size. All models use the identical test row indices. No settings are selected in this module.
