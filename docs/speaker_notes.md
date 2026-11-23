# Presentation Script & Speaker Notes

Estimated duration: 8–10 minutes + 2-minute live demo.

## Slide 1: Predicting Next-Month Credit Card Default

Good afternoon, professors and colleagues. Our project tackles predicting next-month credit card payment defaults using payment history from the UCI repository (30,000 clients). We frame this as an educational binary classification task where predicted default probabilities prioritize accounts for proactive credit review under a 22.1% default imbalance. Our primary ranking metric is Average Precision (AP / PR-AUC). We evaluate four model families under a leak-free StratifiedGroupKFold protocol, training our PyTorch neural network on an NVIDIA GeForce RTX 4050 GPU. Our team divided responsibilities: Yevgeniy Averyanov (SE-2408) led PyTorch CUDA neural modeling and GPU optimization; Madiyar Amanzhol (SE-2407) led classical baseline models, hyperparameter search, and decision threshold selection; and Aldiyar Amangeldi (SE-2407) managed data auditing, StratifiedGroupKFold partitioning, and permutation importance error analysis.

## Slide 2: Problem Formulation & Prediction Boundary

Our dataset comes from the UCI repository (Yeh 2009, CC BY 4.0), capturing 30,000 clients from a major Taiwanese issuer in 2005. Features represent information strictly available by the end of September 2005: demographics, repayment statuses from April to September (PAY_0 through PAY_6), billed amounts, and payment amounts. The target is default in October 2005. ID is strictly an identifier and excluded from all features. Payment status codes -2 and 0 were not documented in the original UCI paper (Yeh 2009). Rather than guessing semantics or imputing, we preserve them as distinct categorical levels.

## Slide 3: Exploratory Data Analysis Informs Preprocessing

Exploratory data analysis revealed three critical characteristics. First, class imbalance: defaults represent 22.1% of training cases. Accuracy is completely misleading because a naive no-default classifier achieves ~78% accuracy while finding zero defaults. Thus, Average Precision is our primary evaluation metric. Second, repayment delay status PAY_0 correlates heavily with immediate default risk. Third, monetary features exhibit extreme positive skewness, zero payments (17.4% in PAY_AMT1), and negative billed balances (409 cases in BILL_AMT1 representing overpayments). We applied a signed logarithm transform followed by StandardScaler fitted strictly on train. Finally, our structural audit found 56 duplicate feature rows, which must not cross partition boundaries.

## Slide 4: Leak-Free Evaluation Protocol & Partitioning

To ensure strict data leakage prevention, we used StratifiedGroupKFold with 20 folds, assigning 14 to train, 3 to validation, and 3 to test. Identical feature vectors were grouped together so that duplicate rows—including those with conflicting labels—never cross between train, validation, and test. All feature transformers learn solely from training data. Validation data is used to tune hyperparameters, pick the early-stopping checkpoint, and choose operating thresholds. The test partition was kept sealed until all decisions were frozen and hashed via SHA-256.

## Slide 5: Documented Hyperparameter Tuning Strategy

Our tuning strategy was declared in advance in config.json to eliminate researcher degrees of freedom. On tabular credit datasets with 23 predictors, extensive grid searching risks overfitting to validation noise. For Logistic Regression, we varied inverse regularization C across {0.1, 1.0, 10.0} (C=0.1 won with val AP 0.5484). For HistGradientBoosting, we evaluated tree capacity and learning rate (lr=0.1, max_leaf_nodes=15 won with val AP 0.5555). For PyTorch MLP on CUDA, we systematically tested capacity (64x32 vs 128x64) against regularization intensity (dropout 0.1 to 0.3, weight decay 1e-4 to 5e-3). Candidate #2 achieved the highest validation AP of 0.5523 by controlling early memorization.

## Slide 6: PyTorch MLP Architecture & Overfitting Analysis

Our neural network features 90 transformed inputs passing through layers of size 64 and 32 with ReLU activations, dropout (0.2), and weight decay (0.001). We trained on GPU using AdamW and unweighted BCEWithLogitsLoss. Looking closely at the learning curves: training AP rises steadily from 0.48 to 0.58+, but validation AP peaks at epoch 17 (0.5523) and plateaus (~0.551). The moderate gap indicates mild overfitting. Early stopping with patience=8 halted training at epoch 25 and restored the optimal epoch 17 weights.

## Slide 7: Final Test Results & Statistical Reliability

This table presents final test results on the 4,500 held-out clients. The validation-selected winner is Gradient Boosting (test AP 0.5662), closely matched by PyTorch MLP (test AP 0.5664). Crucially, 1,000-sample bootstrap 95% confidence intervals for test AP overlap completely: Boosting is [0.536, 0.601] and PyTorch MLP is [0.536, 0.601]. Furthermore, a paired bootstrap test difference between Boosting and MLP shows ΔAP = -0.0004 with 95% CI [-0.013, +0.012], proving they perform identically. Comparing Boosting to Logistic Regression yields a point-estimate gain of +0.011, but the paired bootstrap CI [-0.003, +0.026] crosses zero, meaning nonlinear superiority is not statistically significant at 95% confidence on this finite sample.

## Slide 8: Threshold Trade-Offs & Unsuccessful Experiments

In credit review, selecting the classification threshold is a vital operational decision. At the default threshold of 0.5, our model misses 604 defaults (recall is only 39.3%) because default is a minority class. By maximizing validation F1, our working threshold drops to 0.339: recall jumps to 53.2% (catching 529 defaults) at the cost of increasing false alarms from 195 to 392. We also honestly analyzed our unsuccessful experiments: across our 4 joint candidates, Candidate #2 performed best on validation (0.5523) while Candidate #1 peaked early at epoch 5; expanding to 128x64 in Candidate #3 did not improve over Candidate #2. Because parameters were varied jointly, individual effects cannot be decoupled.

## Slide 9: Model Interpretation & Real Client Error Analysis

Permutation feature importance revealed that PAY_0 (September repayment status) is overwhelmingly the most dominant feature, causing a 0.199 drop in test AP when shuffled due to the recency effect: delinquency in the latest billing cycle is the most direct signal of immediate liquidity trouble. All other features drop <0.01. Examining individual test clients highlights the observational nature of credit modeling: Client #904 was a False Positive—past delays triggered a high risk score of 0.74, but the observed October label was 0; the exact non-default mechanism is unobserved in the data. Client #27520 was a False Negative—a completely clean history gave no warning of default; the trigger occurred outside historical billing records.

## Slide 10: Model Comparison: Why Gradient Boosting Was Selected

Why was Gradient Boosting chosen over PyTorch MLP? Under our strict protocol, model selection was frozen based on validation AP, where Boosting achieved 0.5555 versus MLP 0.5523. On the held-out test set, both models perform virtually identically (0.5662 vs 0.5664, paired bootstrap CI [-0.013, +0.012]). Literature on tabular data (e.g., Grinsztajn et al., 2022) hypothesizes that decision trees possess advantageous inductive biases for tabular features: axis-aligned splits naturally partition discrete ordinal codes like PAY_0 without requiring smooth manifold mappings. Furthermore, our study is subject to critical engineering limitations: single historical 2005 sample, uncalibrated probabilities, and lack of causal interpretability.

## Slide 11: Live Demonstration & Standalone Inference

To demonstrate the pipeline live, we execute our standalone demo script using uv run. It loads the frozen validation-selected Gradient Boosting model and train-fitted preprocessing pipeline, scores unseen held-out test client #0, estimates a default probability of 71.3%, compares it against our working threshold of 0.339, and issues an operational decision to PRIORITIZE FOR REVIEW. Only after the prediction is made does it reveal the ground-truth default label. The CLI is fully modular and ready to evaluate arbitrary new client records.

## Slide 12: References, Sources & Academic AI Disclosure

This slide documents our formal citations and compliance with course academic integrity standards. Our dataset is from Yeh (2009) under CC BY 4.0. We cite empirical literature on tabular deep learning by Grinsztajn et al. (2022). Our entire pipeline is open and reproducible using uv, PyTorch 2.6 CUDA, and scikit-learn. In accordance with university policy, we disclose that LLM assistants were used for script scaffolding and drafting; all models, experiments, CUDA pipelines, and analyses were conducted, verified, and are defended by our student team.

## Slide 13: Thank You & Defense Discussion

Thank you very much for your time and attention. Our complete codebase, data splits, and models are fully reproducible via uv and make. We now welcome any questions regarding our methodology, PyTorch CUDA neural architecture, validation protocol, or error analysis.
