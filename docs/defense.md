# Defense Preparation & Question Guide

## Team Members & Assigned Defense Roles

- **Yevgeniy Averyanov** (SE-2408) — Team Lead. PyTorch MLP architecture, CUDA GPU training pipeline, loss functions & backpropagation, learning curves, and early stopping.
- **Madiyar Amanzhol** (SE-2407) — Classical Machine Learning. Baseline DummyClassifier, Logistic Regression tuning, HistGradientBoosting, validation F1 threshold optimization, and operating trade-offs.
- **Aldiyar Amangeldi** (SE-2407) — Data Engineering & Interpretation. UCI dataset audit, leak-free StratifiedGroupKFold splitting, permutation feature importance (PAY_0 dominance), and concrete client error analysis.

---

## Key Defense Questions & Authoritative Answers

### 1. What exactly does your model predict, and what is the prediction boundary?
We predict the binary next-month default label (`default`): 1 means the client defaults on their payment in October 2005, and 0 means they pay. All predictors strictly capture information available by the end of September 2005 (demographics, payment delay codes `PAY_0` to `PAY_6`, billing amounts, and payments from April to September). We do not predict bankruptcy or legal insolvency, and we do not invent an operational days-past-due rule beyond what the dataset provides.

### 2. Why is Average Precision (AP / PR-AUC) the primary metric instead of Accuracy or ROC-AUC?
In this dataset, default is a minority event with ~22.1% prevalence. A naive majority-class classifier achieves ~78% accuracy by predicting "no default" for every single client, yet it detects zero defaults. 
- **ROC-AUC** can present an overly optimistic picture under class imbalance because the large pool of True Negatives inflates the specificity denominator ($FP + TN$).
- **Average Precision (AP)** summarizes the Precision-Recall curve across all possible score thresholds. It directly penalizes false positive alerts among the prioritized candidates. The baseline for AP with random scoring is equal to the positive prevalence (~0.221).

### 3. What is the difference between Precision and Recall, and how does your threshold affect them?
- **Precision** is $\frac{TP}{TP + FP}$: among all clients prioritized by our model for review, what fraction actually defaults? High precision minimizes wasted review hours on false alarms.
- **Recall** is $\frac{TP}{TP + FN}$: among all actual defaults, what fraction did our model catch? High recall minimizes missed credit losses.
- **Threshold Effect**: At the default threshold of 0.5, our model prioritizes precision (66.7%), but recall is only 39.3% (missing 604 defaults). By selecting the threshold that maximizes validation F1 (0.339), recall increases to 53.2% (catching 529 defaults) while sending 392 false alarms for review. The choice of threshold controls operational workload, not score ranking.

### 4. How did you prevent data leakage during preprocessing and splitting?
We implemented four strict leakage guards:
1. `ID` and the target column were completely removed from feature matrices.
2. We used `StratifiedGroupKFold` (20 folds, assigned 14 train / 3 val / 3 test). Identical raw predictor vectors (56 duplicate rows, including 21 groups with conflicting outcomes) were strictly assigned to the same partition, preventing duplicate profile memorization.
3. All transformers (the signed-log monetary transform, `StandardScaler`, and `OneHotEncoder`) were fitted exclusively on the training partition (`X_train`). Validation and test data were strictly transformed using learned training statistics.
4. Model weights, hyperparameter selections, and decisions were frozen and hashed (SHA-256) in `frozen_selection.json` before loading the test set.

### 5. Why is your split approximately 70/15/15 rather than exact row counts?
Because exact duplicate feature vectors must remain in the same partition. To preserve group integrity without hash collisions, 20 stratified group folds were partitioned 14 / 3 / 3. Ensuring zero leakage between train and test has higher scientific priority than enforcing exact integer splits down to the last decimal place.

### 6. Explain the architecture, loss function, and backpropagation of your PyTorch neural network.
- **Architecture**: A Multi-Layer Perceptron (MLP) with 90 transformed input dimensions (scaled numerics + one-hot categories). It has two hidden layers ($64 \to 32$) with ReLU activation functions, dropout (0.2), and a single linear output logit.
- **Loss Function**: `BCEWithLogitsLoss()`. It combines a sigmoid activation and binary cross-entropy into a single numerically stable layer via the log-sum-exp trick, avoiding floating-point underflow.
- **Backpropagation**:
  1. *Forward pass*: Mini-batches (size 256) compute hidden activations and output logits: $z = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2) + b_3$.
  2. *Loss evaluation*: $L = -\frac{1}{N} \sum [y \log \sigma(z) + (1-y) \log(1 - \sigma(z))]$.
  3. *Backward pass (`loss.backward()`)*: PyTorch uses automatic differentiation (reverse-mode autodiff) and the multivariate chain rule to calculate the partial derivative of the loss with respect to every trainable tensor ($\frac{\partial L}{\partial W_i}, \frac{\partial L}{\partial b_i}$).
  4. *Optimizer step (`optimizer.step()`)*: `AdamW` updates weights using running estimates of first and second gradient moments and applies decoupled weight decay ($10^{-3}$). `optimizer.zero_grad()` clears gradients before the next iteration.

### 7. How did you diagnose and mitigate overfitting in your neural network?
Looking at the learning curves (`artifacts/neural/learning_curves.png`):
- From epoch 1 to 17, both training AP and validation AP rise steadily (validation AP reaches peak **0.5523** at epoch 17).
- Beyond epoch 17, training AP continues rising towards 0.580, whereas validation AP plateaus around ~0.551 and validation loss starts creeping upwards. This divergence indicates mild overfitting.
- Early stopping with a patience of 8 epochs halted execution at epoch 25 and restored the optimal checkpoint from **epoch 17**, ensuring test inference uses generalizable weights.

### 8. What hyperparameter tuning strategy did you use, and why?
We predeclared our tuning candidates in `config.json` before training:
- **Logistic Regression**: We tested $C \in \{0.1, 1.0, 10.0\}$; $C=0.1$ won (val AP 0.5484).
- **HistGradientBoosting**: We tested tree depth/leaves (15 vs 31), learning rate (0.05 vs 0.1), and L2 penalty (1.0 vs 5.0); configuration 2 won (val AP 0.5555).
- **PyTorch MLP**: We evaluated 4 configurations testing capacity ($64 \times 32$ vs $128 \times 64$), learning rate ($10^{-3}$ vs $5 \times 10^{-4}$), dropout ($0.1$ to $0.3$), and weight decay ($10^{-4}$ to $5 \times 10^{-3}$).
- *Why candidate #2 won*: A lower learning rate ($5 \times 10^{-4}$) combined with moderate regularization (dropout 0.2, weight decay $10^{-3}$) allowed Candidate #2 to train stably up to epoch 17, achieving validation AP of **0.5523**.

### 9. Discuss your "unsuccessful experiments" and unexpected findings.
1. **Default 0.50 Threshold Failure**: Using the standard 0.50 probability cutoff failed operationally, missing 60.7% of actual defaults (recall was only 39.3%).
2. **Threshold Operating Trade-Off**: Lowering the threshold to 0.339 caught +136 defaults (+34.6%), but doubled false alarms from 195 to 392, illustrating an operational cost trade-off.
3. **Neural Candidate Search**: Across the 4 jointly varied candidates, Candidate #2 achieved the highest validation AP (0.5523). Candidate #1 peaked prematurely at epoch 5, and expanding capacity to $128 \times 64$ in Candidate #3 did not surpass Candidate #2 (0.5504 vs 0.5523).
4. **Parameter Confounding**: Because width, learning rate, and regularization were varied jointly across candidates, the influence of individual parameters cannot be isolated in isolation.

### 10. Why did Gradient Boosting beat the Neural Network, and are they statistically different?
- **Protocol Decision**: Gradient Boosting was selected strictly because it scored highest on the validation set (val AP 0.5555 vs MLP 0.5523), following our predeclared freeze rule.
- **Statistical Parity on Test**: On the held-out test set, Gradient Boosting achieved test AP **0.5662** and PyTorch MLP achieved **0.5664**. 
  - Bootstrap 95% Confidence Intervals: Boosting **[0.536, 0.601]** vs MLP **[0.536, 0.601]** (overlapping completely).
  - Paired bootstrap test difference: $\Delta AP (\text{GB} - \text{MLP}) = -0.0004$ with 95% CI **[-0.013, +0.012]**. The models perform identically.
  - Comparing GB to Logistic Regression: $\Delta AP (\text{GB} - \text{LR}) = +0.0110$, but the 95% paired bootstrap CI is **[-0.003, +0.026]**, crossing zero. Thus, nonlinear superiority over LR is not statistically significant at alpha=0.05 on this sample.
- **Literature Hypotheses on Tabular Inductive Bias**: Researchers (e.g., Grinsztajn et al., 2022) hypothesize that decision trees naturally form axis-aligned step partitions well-suited for discrete ordinal scales (such as `PAY_0` codes), whereas neural networks require careful continuous smoothing and hyperparameter co-tuning.

### 11. What does Permutation Feature Importance reveal, and why is PAY_0 so dominant?
Permutation importance measures the drop in test AP when an original raw feature's values are shuffled across test records (breaking its relationship with the target without refitting).
- **`PAY_0` (September repayment status)** is overwhelmingly the most dominant feature, causing a **0.1986 drop in test AP** (all other features individually cause $<0.01$ drop).
- *Reason*: Recency effect. The state of repayment in the most recent month before prediction directly reflects active liquidity distress or default momentum. Secondary features like `BILL_AMT1` and `PAY_2` refine the estimate, but `PAY_0` provides the core predictive signal.
- *Caveat*: Permutation importance measures predictive reliance, not causal impact.

### 12. Walk us through specific client errors (False Positives and False Negatives).
From our test error analysis (`artifacts/evaluation/error_examples.csv`):
- **False Positive (Client #904)**: Actual label = 0 (No Default), Model Probability = 0.742 (flagged for review). Client had a 2-month delay in September (`PAY_0 = 2`) with an NT$57k balance. The model flagged them due to acute recent delinquency, but the observed October label was 0; the exact non-default mechanism is unobserved in the data.
- **False Negative (Client #27520)**: Actual label = 1 (Default), Model Probability = 0.080 (not flagged). Client had on-time payments across all 6 prior months (`PAY_0 = 0`, `BILL_AMT1 = NT$89k`). Historical billing records contained zero delinquency warnings; the trigger for default occurred outside the recorded features.
- **True Positive (Client #21027)**: Actual label = 1, Model Probability = 0.726. Repeated delays (`PAY_0 = 2`, `PAY_2 = 3`) and high balance correctly flagged for proactive review.

### 13. How did you handle unusual category codes and negative balances?
- **Unusual Codes**: In `PAY_0` through `PAY_6`, codes `-2` and `0` appear frequently in the data but were omitted from UCI's documentation table (Yeh 2009). Rather than inventing speculative meanings or imputing them, we preserved each code as a distinct level in OneHotEncoding (`handle_unknown="ignore"`).
- **Negative Balances**: In `BILL_AMT1` to `BILL_AMT6`, negative values represent customer overpayments or merchant refunds. We applied a signed-log transform: $f(x) = \text{sign}(x) \cdot \log_{10}(1 + |x|)$, which retains sign and zero while compressing multi-million NT$ outliers, followed by `StandardScaler`.

### 14. What are the key limitations of your study, and what would be needed for production deployment?
1. **Historical & Regional Context**: The data is from Taiwan in 2005. Economic conditions, consumer behavior, and credit products have changed. Validating modern deployment requires current, timestamped cohorts and out-of-time (forward temporal) testing.
2. **Transferability**: Models trained on one bank cannot be deployed at another without external validation for domain shift and data definition mismatches.
3. **Probability Calibration**: While models rank risk effectively (AP ~0.566), raw scores are not calibrated probabilities. Platt scaling or Isotonic Regression on held-out validation data would be required before using scores in financial risk budgeting.
4. **Fairness**: Demographic features (`SEX`, `AGE`, `MARRIAGE`) must undergo demographic parity and equalized odds fairness audits to comply with lending non-discrimination regulations.
5. **No Causality**: Feature importances describe statistical reliance, not levers for financial intervention.

### 15. How do you run the live demonstration?
Execute from the terminal using `uv`:
```powershell
uv run python -m src.demo --index 0 --reveal-label
```
The script loads the serialized validation-selected pipeline (`artifacts/classical/boosting.joblib`), feeds held-out test client #0, displays the estimated default probability (0.7125), compares it with the frozen operating threshold (0.3391), outputs `PRIORITIZE FOR REVIEW`, and reveals the ground-truth outcome (`Actual label: 1`) afterward.
