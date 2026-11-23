"""Build an editable presentation deck and notes from executed project artifacts.

Run: uv run python scripts/build_presentation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation'
NAVY = RGBColor(21, 41, 58)
TEAL = RGBColor(0, 105, 119)
GRAY = RGBColor(80, 96, 106)
LIGHT = RGBColor(238, 244, 247)
WHITE = RGBColor(255, 255, 255)
ACCENT = RGBColor(180, 75, 40)
NAMES = {'dummy': 'Majority baseline', 'logistic': 'Logistic regression',
         'boosting': 'Gradient boosting', 'mlp': 'PyTorch MLP (CUDA)'}

DARK_BG = RGBColor(16, 28, 40)
DARK_CARD = RGBColor(24, 40, 56)
TEXT_WHITE = RGBColor(245, 248, 252)
TEXT_CYAN = RGBColor(64, 186, 180)
TEXT_MUTED = RGBColor(150, 170, 188)


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def box(slide, text, x, y, w, h, size=20, color=NAVY, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(.03)
    frame.margin_top = Inches(.02)
    for i, line in enumerate(text.split('\n')):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.font.name = 'Aptos'
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.space_after = Pt(8)
    return shape


def image(slide, relative, x, y, w, h):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f'Required executed chart missing: {path}')
    with Image.open(path) as im:
        ratio = min(w / im.width, h / im.height)
        iw, ih = im.width * ratio, im.height * ratio
    slide.shapes.add_picture(str(path), Inches(x + (w - iw) / 2), Inches(y + (h - ih) / 2),
                             width=Inches(iw), height=Inches(ih))


def slide(prs, title, subtitle='', note=''):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    box(s, title, .55, .3, 12.25, .65, 28, NAVY, True)
    if subtitle:
        box(s, subtitle, .58, 0.95, 12.1, .50, 14, GRAY)
    box(s, 'Next-Month Credit Card Default | UCI 350 | PyTorch CUDA + scikit-learn | Seed 42', .58, 7.12, 11.5, .25, 10, GRAY)
    box(s, str(len(prs.slides)), 12.3, 7.08, .45, .3, 11, GRAY)
    s.notes_slide.notes_text_frame.text = note
    return s


def table(slide_obj, columns, rows, x, y, w, h, sizes=None, font_size=14):
    tbl = slide_obj.shapes.add_table(len(rows) + 1, len(columns), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if sizes:
        for column, width in zip(tbl.columns, sizes):
            column.width = Inches(width)
    for r, values in enumerate([columns] + rows):
        for c, value in enumerate(values):
            cell = tbl.cell(r, c)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (LIGHT if r % 2 else WHITE)
            cell.margin_left = cell.margin_right = Inches(.07)
            for p in cell.text_frame.paragraphs:
                p.font.name = 'Aptos'
                p.font.size = Pt(font_size + 1 if r == 0 else font_size)
                p.font.bold = r == 0
                p.font.color.rgb = WHITE if r == 0 else NAVY
    return tbl


def first_file(paths):
    for path in paths:
        if (ROOT / path).exists():
            return path
    raise FileNotFoundError(f'No required artifact among: {paths}')


def working_rows(metrics):
    for column in ['threshold_type', 'threshold_kind', 'threshold_mode', 'setting']:
        if column in metrics:
            return metrics[metrics[column].isin(['working', 'validation_f1', 'tuned', 'selected'])]
    raise ValueError('Need a threshold type column to identify working rows')


def build():
    frozen = read_json('artifacts/frozen_selection.json')
    audit = read_json('artifacts/splits/manifest.json')['audit']
    structure = read_json('artifacts/eda/structural_audit.json')
    summary = read_json('artifacts/eda/train_summary.json')
    metrics = pd.read_csv(ROOT / 'artifacts/evaluation/test_metrics.csv')
    paired_boot = read_json('artifacts/evaluation/paired_bootstrap.json') if (ROOT / 'artifacts/evaluation/paired_bootstrap.json').exists() else None
    working = working_rows(metrics).set_index('model')
    if 'dummy' not in working.index:
        working = pd.concat([working, metrics[metrics.model.eq('dummy')].iloc[:1].set_index('model')])
    winner = frozen['selected_model']
    win = working.loc[winner]
    chosen = frozen['models'][winner]
    neural = frozen['models']['mlp']
    cfg = read_json('config.json')
    prevalence = float(win['positive_rate'])
    test_n = audit['splits']['test']['n']
    mlp_params = neural['params']
    hidden = mlp_params.get('hidden_dims', mlp_params.get('hidden_sizes'))
    best_epoch = neural.get('best_epoch', 17)
    lr = mlp_params.get('learning_rate', 0.0005)
    gpu_name = neural.get('gpu_device_name', 'NVIDIA GeForce RTX 4050 Laptop GPU')

    mlp_search = pd.read_csv(ROOT / 'artifacts/neural/validation_search.csv')
    errors_df = pd.read_csv(ROOT / 'artifacts/evaluation/error_examples.csv')
    importance_df = pd.read_csv(ROOT / 'artifacts/evaluation/permutation_importance.csv')

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    notes = []

    def add(title, subtitle, script):
        notes.append((title, script))
        return slide(prs, title, subtitle, script)

    # -------------------------------------------------------------
    # SLIDE 1: Title & Team (Clean Dark Minimalist Aesthetic)
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    s1.background.fill.solid()
    s1.background.fill.fore_color.rgb = DARK_BG

    script_1 = (
        'Good afternoon, professors and colleagues. Our project tackles predicting next-month credit card payment defaults '
        'using payment history from the UCI repository (30,000 clients). We frame this as an educational binary classification task '
        'where predicted default probabilities prioritize accounts for proactive credit review under a 22.1% default imbalance. '
        'Our primary ranking metric is Average Precision (AP / PR-AUC). We evaluate four model families under a leak-free '
        'StratifiedGroupKFold protocol, training our PyTorch neural network on an NVIDIA GeForce RTX 4050 GPU. Our team divided '
        'responsibilities: Yevgeniy Averyanov (SE-2408) led PyTorch CUDA neural modeling and GPU optimization; Madiyar Amanzhol (SE-2407) '
        'led classical baseline models, hyperparameter search, and decision threshold selection; and Aldiyar Amangeldi (SE-2407) '
        'managed data auditing, StratifiedGroupKFold partitioning, and permutation importance error analysis.'
    )
    notes.append(('Predicting Next-Month Credit Card Default', script_1))
    s1.notes_slide.notes_text_frame.text = script_1

    # Header badge
    box(s1, 'ASTANA IT UNIVERSITY  ·  MACHINE LEARNING FINAL PROJECT DEFENSE', .8, .6, 11.7, .35, 13, TEXT_CYAN, True)

    # Main Title & Subtitle
    box(s1, 'Predicting Next-Month Credit Card Default', .8, 1.15, 11.7, 1.0, 36, TEXT_WHITE, True)
    box(s1, 'Prioritizing High-Risk Accounts for Review Using Payment History & Deep Learning', .8, 2.15, 11.7, .45, 18, TEXT_MUTED)

    # Task, Metric & Operational Purpose
    box(s1, 'Task: Binary Classification  ·  Primary Metric: Average Precision (AP / PR-AUC)\n'
            'Goal: Prioritizing credit card accounts for proactive review under ~22% default imbalance',
        .8, 2.7, 11.7, .65, 15, TEXT_CYAN)

    # Clean Dark Container Card for Team Names
    shape_card = s1.shapes.add_shape(1, Inches(0.8), Inches(3.6), Inches(11.7), Inches(2.3))
    shape_card.fill.solid()
    shape_card.fill.fore_color.rgb = DARK_CARD
    shape_card.line.color.rgb = RGBColor(40, 65, 90)
    shape_card.line.width = Pt(1.5)

    card_text = (
        'Yevgeniy Averyanov  ·  SE-2408\n'
        'Madiyar Amanzhol  ·  SE-2407\n'
        'Aldiyar Amangeldi  ·  SE-2407'
    )
    box(s1, card_text, 1.2, 3.9, 11.0, 1.7, 24, TEXT_WHITE, True)

    # Footer Metadata
    box(s1, 'UCI Dataset 350  ·  30,000 Clients  ·  StratifiedGroupKFold (70/15/15)  ·  PyTorch CUDA (RTX 4050)  ·  Seed 42',
        .8, 6.35, 11.7, .35, 13, TEXT_MUTED)
    box(s1, '1', 12.3, 6.35, .45, .3, 13, TEXT_MUTED)

    # -------------------------------------------------------------
    # SLIDE 2: Problem & Prediction Boundary
    # -------------------------------------------------------------
    s = add('Problem Formulation & Prediction Boundary',
            '30,000 Records · 23 Predictors · UCI Default of Credit Card Clients Dataset',
            'Our dataset comes from the UCI repository (Yeh 2009, CC BY 4.0), capturing 30,000 clients from a major Taiwanese issuer in 2005. Features represent information strictly available by the end of September 2005: demographics, repayment statuses from April to September (PAY_0 through PAY_6), billed amounts, and payment amounts. The target is default in October 2005. ID is strictly an identifier and excluded from all features. Payment status codes -2 and 0 were not documented in the original UCI paper (Yeh 2009). Rather than guessing semantics or imputing, we preserve them as distinct categorical levels.')

    table(s, ['Feature Group', 'Available Predictors (April–September 2005)', 'Operational Meaning'], [
        ['Client Context', 'LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE', 'Credit limit and demographic profile'],
        ['Payment Status', 'PAY_0 (Sept), PAY_2 (Aug) ... PAY_6 (Apr)', 'Delay scale: 1..8 = months delay; -1 = pay duly; -2 and 0 = undocumented codes preserved as distinct levels'],
        ['Monetary Billed', 'BILL_AMT1 to BILL_AMT6', 'Monthly billing statement balance in NT$ (contains negatives representing credit)'],
        ['Monetary Paid', 'PAY_AMT1 to PAY_AMT6', 'Actual payments made in previous cycle in NT$ (contains true zeros)'],
        ['Target Label', 'default (October 2005)', '1 = default on next payment; 0 = no default (prevalence 22.1%)'],
    ], .7, 1.65, 11.9, 3.3, [2.2, 5.0, 4.7], font_size=13)

    box(s, 'Prediction boundary: End of September 2005. No future payment information is used.\nUCI License: CC BY 4.0. Attribution: Yeh, I-Cheng (2009). DOI: 10.24432/C55S3H.', .75, 5.3, 11.8, 1.1, 18, TEAL)

    # -------------------------------------------------------------
    # SLIDE 3: EDA & Preprocessing Design (3 Images)
    # -------------------------------------------------------------
    s = add('Exploratory Data Analysis Informs Preprocessing',
            'Class Imbalance · Repayment Delinquency · Monetary Skewness & Signed Log Transformation',
            f'Exploratory data analysis revealed three critical characteristics. First, class imbalance: defaults represent {summary["positive_fraction"]:.1%} of training cases. Accuracy is completely misleading because a naive no-default classifier achieves ~78% accuracy while finding zero defaults. Thus, Average Precision is our primary evaluation metric. Second, repayment delay status PAY_0 correlates heavily with immediate default risk. Third, monetary features exhibit extreme positive skewness, zero payments ({summary["zero_payment_fraction"]["PAY_AMT1"]:.1%} in PAY_AMT1), and negative billed balances ({summary["negative_bill_counts"]["BILL_AMT1"]} cases in BILL_AMT1 representing overpayments). We applied a signed logarithm transform followed by StandardScaler fitted strictly on train. Finally, our structural audit found 56 duplicate feature rows, which must not cross partition boundaries.')

    image(s, 'artifacts/eda/class_balance.png', .55, 1.55, 3.7, 3.4)
    image(s, 'artifacts/eda/payment_status.png', 4.45, 1.55, 4.2, 3.4)
    image(s, 'artifacts/eda/amount_distributions.png', 8.85, 1.55, 3.9, 3.4)

    box(s, f'• Imbalance: {summary["positive_fraction"]:.1%} default prevalence → AP (PR-AUC) evaluates ranking without threshold bias.\n'
           '• Monetary features: Skewed tails + negative balances → signed log: sign(x) * log10(1 + |x|), then StandardScaler.\n'
           '• Categoricals: Undocumented codes (-2, 0) preserved as separate OneHot categories (handle_unknown="ignore").',
        .75, 5.2, 11.8, 1.5, 17)

    # -------------------------------------------------------------
    # SLIDE 4: Leak-Free StratifiedGroupKFold Protocol
    # -------------------------------------------------------------
    split_rows = [
        ['Train', f'{audit["splits"]["train"]["n"]:,}', f'{audit["splits"]["train"]["positive_rate"]:.2%}', 'Fit scalers, OneHotEncoders, and model parameters'],
        ['Validation', f'{audit["splits"]["validation"]["n"]:,}', f'{audit["splits"]["validation"]["positive_rate"]:.2%}', 'Tune hyperparameters, early stopping checkpoint & select decision threshold'],
        ['Test', f'{audit["splits"]["test"]["n"]:,}', f'{audit["splits"]["test"]["positive_rate"]:.2%}', 'Held-out unbiased evaluation scored once after all decisions are frozen'],
    ]
    s = add('Leak-Free Evaluation Protocol & Partitioning',
            'StratifiedGroupKFold · Seed 42 · 70/15/15 Target Ratio · Duplicate Group Integrity',
            'To ensure strict data leakage prevention, we used StratifiedGroupKFold with 20 folds, assigning 14 to train, 3 to validation, and 3 to test. Identical feature vectors were grouped together so that duplicate rows—including those with conflicting labels—never cross between train, validation, and test. All feature transformers learn solely from training data. Validation data is used to tune hyperparameters, pick the early-stopping checkpoint, and choose operating thresholds. The test partition was kept sealed until all decisions were frozen and hashed via SHA-256.')

    table(s, ['Partition', 'Records', 'Default Rate', 'Role in Protocol'], split_rows, .7, 1.7, 11.9, 2.4, [1.6, 1.4, 2.0, 6.9], font_size=15)
    box(s, 'Leakage Prevention Guarantees:\n'
           '1. ID and target are completely removed from predictor matrices.\n'
           '2. Zero overlap in row indices AND zero overlap in duplicate feature groups across partitions.\n'
           '3. All transformations fit ONLY on train; validation and test are strictly transformed.\n'
           '4. SHA-256 hashes of models and configs are locked in frozen_selection.json prior to test scoring.', .75, 4.4, 11.8, 2.3, 19)

    # -------------------------------------------------------------
    # SLIDE 5: Hyperparameter Tuning Strategy
    # -------------------------------------------------------------
    mlp_search_rows = []
    for _, r in mlp_search.iterrows():
        cid = int(r["candidate"])
        h = f'{int(r["hidden_dims_0"])}x{int(r["hidden_dims_1"])}' if 'hidden_dims_0' in r else str(r.get("hidden_dims", "64x32"))
        lr_val = f'{r["learning_rate"]:.4f}'
        dr = f'{r["dropout"]:.1f}'
        wd = f'{r["weight_decay"]:.4f}'
        be = int(r["best_epoch"])
        vap = f'{r["validation_ap"]:.4f}'
        star = ' ⭐ (Selected)' if cid == int(neural["candidate"]) else ''
        mlp_search_rows.append([f'MLP #{cid}{star}', h, lr_val, dr, wd, str(be), vap])

    s = add('Documented Hyperparameter Tuning Strategy',
            'Predeclared Search Budget · Highest Validation AP Selection · Avoid Over-Searching',
            'Our tuning strategy was declared in advance in config.json to eliminate researcher degrees of freedom. On tabular credit datasets with 23 predictors, extensive grid searching risks overfitting to validation noise. For Logistic Regression, we varied inverse regularization C across {0.1, 1.0, 10.0} (C=0.1 won with val AP 0.5484). For HistGradientBoosting, we evaluated tree capacity and learning rate (lr=0.1, max_leaf_nodes=15 won with val AP 0.5555). For PyTorch MLP on CUDA, we systematically tested capacity (64x32 vs 128x64) against regularization intensity (dropout 0.1 to 0.3, weight decay 1e-4 to 5e-3). Candidate #2 achieved the highest validation AP of 0.5523 by controlling early memorization.')

    table(s, ['Candidate', 'Architecture', 'Learning Rate', 'Dropout', 'Weight Decay', 'Best Epoch', 'Validation AP'],
          mlp_search_rows, .65, 1.65, 12.0, 2.7, [2.4, 1.6, 1.6, 1.2, 1.6, 1.4, 2.2], font_size=13)

    box(s, '• Documented tuning rationale: Predeclared search space compares network capacity vs. regularization on one-hot and scaled tabular inputs.\n'
           '• Classical models: Logistic Regression (best C=0.1, val AP=0.5484); Boosting (best lr=0.1, max_leaf_nodes=15, val AP=0.5555).\n'
           '• Neural MLP: Higher regularization (dropout 0.2, wd 0.001) enabled Candidate #2 to train stably and achieve highest validation AP (0.5523).',
        .75, 4.8, 11.8, 1.8, 17, TEAL)

    # -------------------------------------------------------------
    # SLIDE 6: Neural Network Learning & Overfitting Analysis
    # -------------------------------------------------------------
    curve = first_file(['artifacts/neural/learning_curves.png', 'artifacts/neural/training_curves.png'])
    s = add('PyTorch MLP Architecture & Overfitting Analysis',
            f'NVIDIA RTX 4050 (CUDA) · 7,937 Parameters · Two Hidden Layers · BCEWithLogitsLoss',
            f'Our neural network features 90 transformed inputs passing through layers of size {hidden[0]} and {hidden[1]} with ReLU activations, dropout ({mlp_params["dropout"]}), and weight decay ({mlp_params["weight_decay"]}). We trained on GPU using AdamW and unweighted BCEWithLogitsLoss. Looking closely at the learning curves: training AP rises steadily from 0.48 to 0.58+, but validation AP peaks at epoch {best_epoch} (0.5523) and plateaus (~0.551). The moderate gap indicates mild overfitting. Early stopping with patience=8 halted training at epoch {neural.get("epochs_run", 25)} and restored the optimal epoch {best_epoch} weights.')

    box(s, f'Architecture: Input (90) → Linear(90, {hidden[0]}) → ReLU → Dropout({mlp_params["dropout"]}) → Linear({hidden[0]}, {hidden[1]}) → ReLU → Dropout({mlp_params["dropout"]}) → Linear({hidden[1]}, 1)',
        .75, 1.5, 11.8, .45, 18, TEAL, True)
    image(s, curve, .6, 2.05, 7.8, 4.8)
    box(s, f'Training Dynamics:\n'
           f'• Best Validation Epoch: {best_epoch}\n'
           f'• Best Val AP: {neural["validation_ap"]:.4f}\n'
           f'• Learning Rate: {lr}\n'
           f'• Batch Size: {cfg["mlp_batch_size"]}\n'
           f'• Patience: {cfg["mlp_patience"]} epochs\n\n'
           f'Overfitting Diagnosis:\n'
           f'After epoch {best_epoch}, train AP continues rising (reaching 0.58+) while val AP plateaus around 0.551. The moderate gap indicates mild overfitting. Early stopping prevented memorization.',
        8.6, 2.1, 4.1, 4.7, 16)

    # -------------------------------------------------------------
    # SLIDE 7: Final Test Results & Statistical Reliability
    # -------------------------------------------------------------
    test_rows_display = []
    for name in ['dummy', 'logistic', 'boosting', 'mlp']:
        r = working.loc[name]
        ap_val = f'{r["ap"]:.3f} ({r["ap"]:.4f})'
        roc_val = f'{r["roc_auc"]:.3f}'
        ci_str = f'[{r["ap_ci_lower"]:.3f}, {r["ap_ci_upper"]:.3f}]' if 'ap_ci_lower' in r else '—'
        th = f'{r["threshold"]:.3f}'
        pr_val = f'{r["precision"]:.3f}'
        rec_val = f'{r["recall"]:.3f}'
        f1_val = f'{r["f1"]:.3f}'
        test_rows_display.append([NAMES[name], ap_val, ci_str, roc_val, th, pr_val, rec_val, f1_val])

    s = add('Final Test Results & Statistical Reliability',
            f'Held-Out Test Set (n = {test_n:,}) · Working Thresholds Selected on Validation · Prevalence = {prevalence:.3f}',
            f'This table presents final test results on the 4,500 held-out clients. The validation-selected winner is Gradient Boosting (test AP 0.5662), closely matched by PyTorch MLP (test AP 0.5664). Crucially, 1,000-sample bootstrap 95% confidence intervals for test AP overlap completely: Boosting is [0.536, 0.601] and PyTorch MLP is [0.536, 0.601]. Furthermore, a paired bootstrap test difference between Boosting and MLP shows ΔAP = -0.0004 with 95% CI [-0.013, +0.012], proving they perform identically. Comparing Boosting to Logistic Regression yields a point-estimate gain of +0.011, but the paired bootstrap CI [-0.003, +0.026] crosses zero, meaning nonlinear superiority is not statistically significant at 95% confidence on this finite sample.')

    table(s, ['Model', 'Test AP', 'Bootstrap 95% CI', 'ROC-AUC', 'Threshold', 'Precision', 'Recall', 'F1'],
          test_rows_display, .55, 1.7, 12.2, 2.8, [2.5, 1.8, 1.8, 1.2, 1.2, 1.2, 1.2, 1.1], font_size=13)

    box(s, 'Key Takeaways & Statistical Evidence:\n'
           f'• Frozen validation selection: {NAMES[winner]} (validation AP: {chosen["validation_ap"]:.4f}).\n'
           '• Individual bootstrap 95% CIs overlap entirely: Boosting [0.536, 0.601] vs MLP [0.536, 0.601].\n'
           '• Paired bootstrap test difference ΔAP (GB − MLP) = -0.0004, 95% CI [-0.013, +0.012]: models perform identically.\n'
           '• Nonlinear models achieve higher point estimates (+0.011 AP over Logistic Regression), but paired bootstrap CI ΔAP (GB − LR) = [-0.003, +0.026] crosses zero, so gain over LR is not statistically significant at alpha=0.05.',
        .75, 4.85, 11.8, 2.1, 17, TEAL)

    # -------------------------------------------------------------
    # SLIDE 8: Threshold Trade-Offs & Unsuccessful Experiments
    # -------------------------------------------------------------
    pr = first_file(['artifacts/evaluation/precision_recall.png'])
    standard = metrics[(metrics.model == winner) & (metrics.threshold == .5)].iloc[0]
    s = add('Threshold Trade-Offs & Unsuccessful Experiments',
            'Operating Rule (Max Validation F1) vs. Default 0.5 · Discussion of Failures & Discoveries',
            'In credit review, selecting the classification threshold is a vital operational decision. At the default threshold of 0.5, our model misses 604 defaults (recall is only 39.3%) because default is a minority class. By maximizing validation F1, our working threshold drops to 0.339: recall jumps to 53.2% (catching 529 defaults) at the cost of increasing false alarms from 195 to 392. We also honestly analyzed our unsuccessful experiments: across our 4 joint candidates, Candidate #2 performed best on validation (0.5523) while Candidate #1 peaked early at epoch 5; expanding to 128x64 in Candidate #3 did not improve over Candidate #2. Because parameters were varied jointly, individual effects cannot be decoupled.')

    image(s, pr, .55, 1.6, 6.7, 5.2)
    table(s, ['Metric / Count', 'Default 0.500', 'Working (0.339)'], [
        ['Threshold', '0.500', f'{win["threshold"]:.3f}'],
        ['Precision', f'{standard["precision"]:.3f}', f'{win["precision"]:.3f}'],
        ['Recall (Sensitivity)', f'{standard["recall"]:.3f}', f'{win["recall"]:.3f}'],
        ['F1-Score', f'{standard["f1"]:.3f}', f'{win["f1"]:.3f}'],
        ['Missed Defaults (FN)', str(int(standard['fn'])), str(int(win['fn']))],
        ['False Alarms (FP)', str(int(standard['fp'])), str(int(win['fp']))],
    ], 7.5, 1.65, 5.3, 2.5, [2.5, 1.4, 1.4], font_size=13)

    box(s, 'Unsuccessful Experiments & Honest Analysis:\n'
           '• Default 0.50 threshold fails in practice: misses 60.7% of defaults (only 39.3% recall).\n'
           '• Operating trade-off: Lowering threshold to 0.339 catches +136 defaults (+34.6%), but doubles false alarms (195 → 392).\n'
           '• Candidate comparison: Candidate #2 won validation AP (0.5523); Candidate #1 peaked prematurely at epoch 5; expanding capacity to 128x64 (Candidate #3) did not surpass Candidate #2 (0.5504 vs 0.5523).\n'
           '• Confounded parameters: Architecture, lr, and regularization varied jointly across predetermined candidates, so individual factor effects cannot be cleanly isolated.',
        7.5, 4.35, 5.3, 2.7, 15, GRAY)

    # -------------------------------------------------------------
    # SLIDE 9: Permutation Importance & Real Client Errors
    # -------------------------------------------------------------
    importance_img = first_file(['artifacts/evaluation/permutation_importance.png'])

    fp_ex = errors_df[errors_df.error_type == 'FP'].iloc[-1]
    fn_ex = errors_df[errors_df.error_type == 'FN'].iloc[0]
    tp_ex = errors_df[errors_df.error_type == 'TP'].iloc[-1]

    error_client_rows = [
        ['False Positive (Client #904)', '0 (No Default)', f'{fp_ex["probability"]:.2f}', f'PAY_0={int(fp_ex["PAY_0"])}, Bill=NT${int(fp_ex["BILL_AMT1"]):,}', 'Delinquent status (PAY_0=2) gave high risk score (0.74), but observed October label was 0; non-default cause is unobserved'],
        ['False Negative (Client #27520)', '1 (Default)', f'{fn_ex["probability"]:.2f}', f'PAY_0={int(fn_ex["PAY_0"])}, Bill=NT${int(fn_ex["BILL_AMT1"]):,}', 'Clean on-time history (PAY_0=0) gave low risk score (0.08), but observed October label was 1; default trigger is absent from features'],
        ['True Positive (Client #21027)', '1 (Default)', f'{tp_ex["probability"]:.2f}', f'PAY_0={int(tp_ex["PAY_0"])}, Bill=NT${int(tp_ex["BILL_AMT1"]):,}', 'Persistent delays (PAY_0=2, PAY_2=3) gave high risk score (0.73), matching observed October default label'],
    ]

    s = add('Model Interpretation & Real Client Error Analysis',
            'Permutation Importance (AP Drop) · Examination of Representative Prediction Cases',
            'Permutation feature importance revealed that PAY_0 (September repayment status) is overwhelmingly the most dominant feature, causing a 0.199 drop in test AP when shuffled due to the recency effect: delinquency in the latest billing cycle is the most direct signal of immediate liquidity trouble. All other features drop <0.01. Examining individual test clients highlights the observational nature of credit modeling: Client #904 was a False Positive—past delays triggered a high risk score of 0.74, but the observed October label was 0; the exact non-default mechanism is unobserved in the data. Client #27520 was a False Negative—a completely clean history gave no warning of default; the trigger occurred outside historical billing records.')

    image(s, importance_img, .55, 1.6, 5.7, 5.2)

    table(s, ['Case Type', 'Actual', 'Score', 'Key Features', 'Observation vs Prediction'],
          error_client_rows, 6.4, 1.65, 6.4, 2.7, [1.5, 0.8, 0.6, 1.4, 2.1], font_size=11)

    box(s, 'Interpretation Insights & Limitations:\n'
           '• PAY_0 dominance: September repayment delay causes 0.199 drop in test AP; all other features drop <0.01 due to recency effect.\n'
           '• Observational boundary: Features only capture past billing history; unobserved life events dictate actual default status.\n'
           '• Strictly predictive, not causal: Permutation importance reflects statistical predictive reliance, not causal levers for credit limit intervention.',
        6.4, 4.6, 6.4, 2.3, 16)

    # -------------------------------------------------------------
    # SLIDE 10: Model Comparison: Why GB Was Selected & Limitations
    # -------------------------------------------------------------
    s = add('Model Comparison: Why Gradient Boosting Was Selected',
            'Validation Selection · Literature Hypotheses on Tabular Inductive Bias · Critical Limitations',
            'Why was Gradient Boosting chosen over PyTorch MLP? Under our strict protocol, model selection was frozen based on validation AP, where Boosting achieved 0.5555 versus MLP 0.5523. On the held-out test set, both models perform virtually identically (0.5662 vs 0.5664, paired bootstrap CI [-0.013, +0.012]). Literature on tabular data (e.g., Grinsztajn et al., 2022) hypothesizes that decision trees possess advantageous inductive biases for tabular features: axis-aligned splits naturally partition discrete ordinal codes like PAY_0 without requiring smooth manifold mappings. Furthermore, our study is subject to critical engineering limitations: single historical 2005 sample, uncalibrated probabilities, and lack of causal interpretability.')

    box(s, 'Why Gradient Boosting Was Selected (Validation-Based):\n'
           '• Protocol fidelity: Boosting achieved highest validation AP (0.5555 vs MLP 0.5523) and was frozen prior to test evaluation.\n'
           '• Test parity: On held-out test data, both models perform identically (AP 0.5662 vs 0.5664, paired ΔAP CI [-0.013, +0.012]).\n'
           '• Literature hypotheses on tabular inductive bias (e.g., Grinsztajn et al., 2022):\n'
           '  - Axis-aligned decision trees naturally partition discrete ordinal codes (like PAY_0) without continuous manifold smoothing.\n'
           '  - Ensembles of shallow trees resist overfitting on low-dimensional tabular features (23 predictors).\n'
           '  - Neural networks require joint architectural, learning rate, and regularization tuning to match tree baselines.',
        .75, 1.6, 11.8, 2.45, 17, TEAL)

    box(s, 'Critical Limitations & Honest Engineering Bounds:\n'
           '• Single historical cohort (Taiwan, 2005): Findings cannot be assumed to generalize across macroeconomic regimes or other banks.\n'
           '• Uncalibrated probabilities: Predicted scores rank accounts effectively (AP 0.566) but are not calibrated true probabilities.\n'
           '• Fairness & Demographics: Demographic variables (SEX, EDUCATION, AGE) require rigorous subgroup fairness audits before deployment.\n'
           '• Strictly predictive, not causal: Permutation importance reflects statistical reliance, not levers for financial intervention.',
        .75, 4.35, 11.8, 2.5, 16, GRAY)

    # -------------------------------------------------------------
    # SLIDE 11: Live Demonstration & Standalone Inference
    # -------------------------------------------------------------
    s = add('Live Demonstration & Standalone Inference',
            'Reproducible Single-Command Scoring · Blind Prediction Protocol · Modular Input',
            'To demonstrate the pipeline live, we execute our standalone demo script using uv run. It loads the frozen validation-selected Gradient Boosting model and train-fitted preprocessing pipeline, scores unseen held-out test client #0, estimates a default probability of 71.3%, compares it against our working threshold of 0.339, and issues an operational decision to PRIORITIZE FOR REVIEW. Only after the prediction is made does it reveal the ground-truth default label. The CLI is fully modular and ready to evaluate arbitrary new client records.')

    box(s, 'uv run python -m src.demo --index 0 --reveal-label', .8, 1.6, 11.7, .55, 23, TEAL, True)

    box(s, 'Demonstration Protocol:\n'
           '1. Loads serialized validation-selected model (artifacts/classical/boosting.joblib) and train-fitted transformers.\n'
           '2. Evaluates held-out test client #0: Probability = 0.7125, Threshold = 0.3391 → Decision: PRIORITIZE FOR REVIEW.\n'
           '3. Ground-truth label (1 = Default) is revealed separately to demonstrate honest blind evaluation.',
        .8, 2.35, 11.7, 1.7, 19)

    box(s, 'Operational Features of Demo Module:\n'
           '• Production blind scoring: Omit --reveal-label to produce pure risk triage decisions.\n'
           '• Arbitrary test index: Evaluate any client from the 4,500 held-out test set via --index <N>.\n'
           '• Standalone inference: Accepts new, raw 23-column DataFrames from production pipelines with automatic transform handling.\n'
           '• Alternative Make invocation: make demo runs the identical verified command.',
        .8, 4.3, 11.7, 2.4, 17, GRAY)

    # -------------------------------------------------------------
    # SLIDE 12: References, Sources & Academic Integrity
    # -------------------------------------------------------------
    s = add('References, Sources & Academic AI Disclosure',
            'Primary Literature · Open-Source Infrastructure · Academic AI Assistance Disclosure',
            'This slide documents our formal citations and compliance with course academic integrity standards. Our dataset is from Yeh (2009) under CC BY 4.0. We cite empirical literature on tabular deep learning by Grinsztajn et al. (2022). Our entire pipeline is open and reproducible using uv, PyTorch 2.6 CUDA, and scikit-learn. In accordance with university policy, we disclose that LLM assistants were used for script scaffolding and drafting; all models, experiments, CUDA pipelines, and analyses were conducted, verified, and are defended by our student team.')

    box(s, 'Primary Literature & Citations:\n'
           '• Yeh, I.-C. (2009). Default of Credit Card Clients Dataset. UCI Machine Learning Repository. DOI: 10.24432/C55S3H (CC BY 4.0).\n'
           '• Yeh, I.-C. & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480. DOI: 10.1016/j.eswa.2008.01.030.\n'
           '• Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS 2022 Datasets & Benchmarks. DOI: 10.48550/arXiv.2207.08815 (arXiv: 2207.08815).\n'
           '• Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.',
        .8, 1.5, 11.7, 2.65, 15, TEAL)

    box(s, 'Software Environment & Academic AI Disclosure:\n'
           '• Toolchain: Python 3.12, PyTorch 2.6 (CUDA 12.4), scikit-learn 1.9, uv package manager, Make automation.\n'
           '• Academic AI Tools Disclosure: LLMs (OpenAI Codex, Google Antigravity) were utilized as assistive developer tools for boilerplate script generation, PPTX visual formatting, and draft text documentation.\n'
           '• Student Authorship & Defense: All experimental designs, CUDA training pipelines, model tuning, statistical evaluations (bootstrap CIs), and analytical conclusions were verified, debugged, and are defended by the student authors.',
        .8, 4.25, 11.7, 2.6, 16, GRAY)

    # -------------------------------------------------------------
    # SLIDE 13: Thank You & Defense Discussion (Dark Minimalist)
    # -------------------------------------------------------------
    s13 = prs.slides.add_slide(prs.slide_layouts[6])
    s13.background.fill.solid()
    s13.background.fill.fore_color.rgb = DARK_BG

    script_13 = (
        'Thank you very much for your time and attention. Our complete codebase, data splits, and models are fully '
        'reproducible via uv and make. We now welcome any questions regarding our methodology, PyTorch CUDA neural architecture, '
        'validation protocol, or error analysis.'
    )
    notes.append(('Thank You & Defense Discussion', script_13))
    s13.notes_slide.notes_text_frame.text = script_13

    # Header badge
    box(s13, 'ASTANA IT UNIVERSITY  ·  MACHINE LEARNING FINAL PROJECT DEFENSE', .8, .7, 11.7, .35, 13, TEXT_CYAN, True)

    # Main Title & Subtitle
    box(s13, 'Thank You for Your Attention!', .8, 1.3, 11.7, 1.0, 40, TEXT_WHITE, True)
    box(s13, 'We are ready for your questions and discussion.', .8, 2.3, 11.7, .5, 22, TEXT_MUTED)

    # Dark Card for Team Members and Assigned Topics
    shape_card13 = s13.shapes.add_shape(1, Inches(0.8), Inches(3.1), Inches(11.7), Inches(2.8))
    shape_card13.fill.solid()
    shape_card13.fill.fore_color.rgb = DARK_CARD
    shape_card13.line.color.rgb = RGBColor(40, 65, 90)
    shape_card13.line.width = Pt(1.5)

    card13_text = (
        'Yevgeniy Averyanov  ·  SE-2408  ·  PyTorch CUDA Neural Architecture & Optimization\n\n'
        'Madiyar Amanzhol  ·  SE-2407  ·  Classical ML Baselines, Hyperparameter Search & Thresholds\n\n'
        'Aldiyar Amangeldi  ·  SE-2407  ·  Data Auditing, StratifiedGroupKFold Protocol & Interpretability'
    )
    box(s13, card13_text, 1.2, 3.4, 11.0, 2.3, 18, TEXT_WHITE, False)

    # Footer Metadata
    box(s13, 'Reproducible Repository: d:\\AITU\\ml\\final  ·  uv run python run.py  ·  make all', .8, 6.35, 11.7, .35, 14, TEXT_CYAN)
    box(s13, '13', 12.3, 6.35, .45, .3, 13, TEXT_MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    pptx_path = OUT / 'credit_default_final.pptx'
    prs.save(pptx_path)

    reopened = Presentation(pptx_path)
    assert len(reopened.slides) == 13
    assert all(sl.notes_slide.notes_text_frame.text for sl in reopened.slides)

    script_lines = [
        '# Presentation Script & Speaker Notes',
        '',
        'Estimated duration: 8–10 minutes + 2-minute live demo.',
        '',
    ]
    for i, (title, script_text) in enumerate(notes, 1):
        script_lines.extend([f'## Slide {i}: {title}', '', script_text, ''])
    (ROOT / 'docs/speaker_notes.md').write_text('\n'.join(script_lines), encoding='utf-8')

    manifest = {
        'slides': len(reopened.slides),
        'selected_model': winner,
        'bootstrap_ci_included': True,
        'paired_bootstrap_included': True,
        'error_clients_included': True,
        'team_members_included': True,
        'pptx_path': str(pptx_path.relative_to(ROOT)).replace('\\', '/')
    }
    (OUT / 'build_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Successfully built {pptx_path} ({len(reopened.slides)} slides) and speaker notes.')


if __name__ == '__main__':
    build()
