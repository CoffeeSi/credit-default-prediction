# Sources and attribution

- Course requirements: `Machine_Learning_Final_Project_Instructions.pdf`.
- Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H . Dataset page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients . License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ . Downloaded from UCI; target renamed to `default` and XLS converted to CSV. Attribution and original raw files are retained.
- Yeh, I.-C. and Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track, 507–520. https://doi.org/10.52202/068431-0037 . Preprint: arXiv:2207.08815.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems 32 (NeurIPS 2019). arXiv:1912.01703.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
- scikit-learn AP definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html . Average Precision summarizes the precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
- PyTorch BCEWithLogitsLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html . Numerically stable combination of sigmoid and binary cross entropy.
- uv package manager: https://docs.astral.sh/uv/ . Used for modern, reproducible Python 3.12 virtual environment management and package locking.

# Project Team & Individual Contributions

- **Yevgeniy Averyanov** (SE-2408) — Team Lead. Responsible for PyTorch neural network design (MLP architecture, BCEWithLogitsLoss, AdamW optimizer, CUDA GPU pipeline), learning curve analysis, early stopping implementation, and overall pipeline integration.
- **Madiyar Amanzhol** (SE-2407) — Classical Machine Learning. Responsible for baseline DummyClassifier, Logistic Regression (lbfgs with L2 regularization), HistGradientBoosting tuning, threshold selection rule via validation F1 maximization, and operating tradeoff analysis.
- **Aldiyar Amangeldi** (SE-2407) — Data Engineering & Model Interpretation. Responsible for UCI dataset schema verification, leak-free StratifiedGroupKFold partitioning (preserving duplicate feature rows), permutation importance evaluation (identifying PAY_0 dominance), and concrete client error analysis.

# AI Assistance Disclosure

In accordance with course academic integrity guidelines:
- LLM coding agents were utilized as assistive developer tools to accelerate boilerplate script generation, data schema audits, formatting of presentation slides, and draft text generation.
- All machine learning algorithms, architectural choices, validation rules, loss functions, statistical analyses (including 1,000-sample bootstrap confidence intervals), and analytical interpretations were designed, executed, reviewed, and defended directly by the student authors.
- All numbers reported in the presentation, README, and defense documentation strictly reflect executed results from verified disk artifacts.
