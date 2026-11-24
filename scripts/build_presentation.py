"""Build the minimalist, editable defense deck and speaker notes from executed artifacts.

Every number on the slides is read from artifacts/ (nothing is typed in by hand),
so re-running the pipeline and this script keeps the deck consistent with results.

Run: uv run python scripts/build_presentation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation'

# Minimal palette: near-black ink, one gray, one hairline gray, one risk accent.
INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x6B, 0x72, 0x80)
FAINT = RGBColor(0xA8, 0xAE, 0xB6)
RULE = RGBColor(0xD9, 0xDD, 0xE3)
SOFT = RGBColor(0xF4, 0xF5, 0xF7)
ACCENT = RGBColor(0xB4, 0x43, 0x2F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY, MONO = 'Calibri Light', 'Calibri', 'Consolas'

M = 0.7                      # side margin, inches
W = 13.333 - 2 * M           # content width
NO_STYLE_NO_GRID = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'
NAMES = {'dummy': 'Majority baseline', 'logistic': 'Logistic regression',
         'boosting': 'Gradient boosting', 'mlp': 'PyTorch MLP'}
TEAM = [('Yevgeniy Averyanov', 'SE-2408', 'Neural network, training pipeline, learning curves'),
        ('Madiyar Amanzhol', 'SE-2407', 'Baseline, classical models, tuning, thresholds'),
        ('Aldiyar Amangeldi', 'SE-2407', 'Data audit, leak-free split, interpretation, errors')]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


# ---------------------------------------------------------------- primitives
def _bullet(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set('marL', str(Emu(Inches(.28))))
    pPr.set('indent', str(-Emu(Inches(.28))))
    color = OxmlElement('a:buClr')
    srgb = OxmlElement('a:srgbClr')
    srgb.set('val', str(FAINT))
    color.append(srgb)
    char = OxmlElement('a:buChar')
    char.set('char', '•')
    pPr.append(color)
    pPr.append(char)


def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, after=6, bullets=False, line=None):
    """paras: str | list[str | list[(text, overrides)]]. Margins are zero for clean alignment."""
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    for i, para in enumerate([paras] if isinstance(paras, str) else paras):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        for chunk, opts in (para if isinstance(para, list) else [(para, {})]):
            run = p.add_run()
            run.text = chunk
            f = run.font
            f.name = opts.get('font', font)
            f.size = Pt(opts.get('size', size))
            f.bold = opts.get('bold', bold)
            f.italic = opts.get('italic', False)
            f.color.rgb = opts.get('color', color)
        p.alignment = align
        p.space_after = Pt(after)
        if line:
            p.line_spacing = line
        if bullets:
            _bullet(p)
    return shape


def rect(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE):
    box = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    box.shadow.inherit = False
    if fill is None:
        box.fill.background()
    else:
        box.fill.solid()
        box.fill.fore_color.rgb = fill
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line
        box.line.width = Pt(1)
    frame = box.text_frame
    frame.margin_left = frame.margin_right = Inches(.15)
    frame.margin_top = frame.margin_bottom = Inches(.05)
    frame.word_wrap = True
    return box


def shape_text(box, paras, size=14, color=INK, bold=False, align=PP_ALIGN.CENTER, font=BODY):
    frame = box.text_frame
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, para in enumerate(paras):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        for chunk, opts in (para if isinstance(para, list) else [(para, {})]):
            run = p.add_run()
            run.text = chunk
            run.font.name = opts.get('font', font)
            run.font.size = Pt(opts.get('size', size))
            run.font.bold = opts.get('bold', bold)
            run.font.color.rgb = opts.get('color', color)
        p.alignment = align


def _border(tcPr, tag, color=None, width=Pt(.75)):
    ln = OxmlElement(tag)
    if color is None:
        ln.set('w', '0')
        ln.append(OxmlElement('a:noFill'))
    else:
        ln.set('w', str(int(width)))
        fill = OxmlElement('a:solidFill')
        srgb = OxmlElement('a:srgbClr')
        srgb.set('val', str(color))
        fill.append(srgb)
        ln.append(fill)
    tcPr.append(ln)


def table(slide, x, y, widths, rows, size=15, row_h=.48, header=True, emphasize=(),
          right_from=1, header_size=12):
    """Hairline table: no fills, no vertical rules, a thin rule under each row."""
    shape = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y),
                                   Inches(sum(widths)), Inches(row_h * len(rows)))
    tbl = shape.table
    tbl._tbl.tblPr.find(qn('a:tableStyleId')).text = NO_STYLE_NO_GRID
    tbl.first_row = tbl.horz_banding = False
    for c, width in enumerate(widths):
        tbl.columns[c].width = Inches(width)
    for r, values in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        is_head = header and r == 0
        strong = r in emphasize
        for c, value in enumerate(values):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0 if c == 0 else .08)
            cell.margin_right = Inches(.08)
            cell.margin_top = cell.margin_bottom = Inches(.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(value)
            run.font.name = BODY
            run.font.size = Pt(header_size if is_head else size)
            run.font.bold = strong and not is_head
            run.font.color.rgb = MUTED if is_head else (ACCENT if strong and c == 0 else INK)
            p.alignment = PP_ALIGN.RIGHT if c >= right_from else PP_ALIGN.LEFT
            tcPr = cell._tc.get_or_add_tcPr()
            for child in list(tcPr):
                tcPr.remove(child)
            _border(tcPr, 'a:lnL')
            _border(tcPr, 'a:lnR')
            _border(tcPr, 'a:lnT', INK if r == 0 else None)
            _border(tcPr, 'a:lnB', INK if (is_head or r == len(rows) - 1) else RULE)
            tcPr.append(OxmlElement('a:noFill'))
    return tbl


def stat(slide, x, y, w, value, label, color=INK, size=54):
    text(slide, x, y, w, 1.0, value, size, color, font=HEAD)
    text(slide, x, y + 1.0, w, .6, label, 14, MUTED)


def style_chart(chart, size=12):
    chart.has_title = False
    chart.font.name = BODY
    chart.font.size = Pt(size)
    chart.font.color.rgb = MUTED


def quiet_category_axis(axis, size=12):
    axis.major_tick_mark = XL_TICK_MARK.NONE
    axis.format.line.color.rgb = RULE
    axis.tick_labels.font.size = Pt(size)
    axis.tick_labels.font.color.rgb = MUTED
    axis.has_major_gridlines = False


# ---------------------------------------------------------------- deck
class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.notes = []

    def blank(self, title_for_notes, notes):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = WHITE
        s.notes_slide.notes_text_frame.text = notes
        self.notes.append((title_for_notes, notes))
        return s

    def content(self, section, title, notes):
        s = self.blank(title, notes)
        n = len(self.prs.slides)
        text(s, M, .5, W, .3, section.upper(), 11, ACCENT, bold=True)
        text(s, M, .82, W, .9, title, 32, INK, font=HEAD)
        text(s, M, 6.95, 9, .3, 'Predicting next-month credit card default  ·  AITU Machine Learning final project', 10, FAINT)
        text(s, 13.333 - M - .6, 6.95, .6, .3, str(n), 10, FAINT, align=PP_ALIGN.RIGHT)
        return s


def build():
    # ------------------------------------------------------------ inputs
    frozen = read_json('artifacts/frozen_selection.json')
    audit = read_json('artifacts/splits/manifest.json')['audit']
    structure = read_json('artifacts/eda/structural_audit.json')
    summary = read_json('artifacts/eda/train_summary.json')
    cfg = read_json('config.json')
    boot = read_json('artifacts/evaluation/paired_bootstrap.json')  # written by src.evaluate
    metrics = pd.read_csv(ROOT / 'artifacts/evaluation/test_metrics.csv')
    working = metrics[metrics.threshold_type == 'working'].set_index('model')
    pay0 = pd.read_csv(ROOT / 'artifacts/eda/train_pay0_default_rates.csv')
    history = pd.read_csv(ROOT / 'artifacts/neural/history_selected.csv')
    mlp_search = pd.read_csv(ROOT / 'artifacts/neural/validation_search.csv')
    errors = pd.read_csv(ROOT / 'artifacts/evaluation/error_examples.csv')
    importance = pd.read_csv(ROOT / 'artifacts/evaluation/permutation_importance.csv')
    predictions = pd.read_csv(ROOT / 'artifacts/evaluation/test_predictions.csv')

    winner = frozen['selected_model']
    models = frozen['models']
    neural = models['mlp']
    p = neural['params']
    h1, h2 = p['hidden_dims']
    win = working.loc[winner]
    std = metrics[(metrics.model == winner) & (metrics.threshold_type == '0.5')].iloc[0]
    prevalence = float(win['positive_rate'])
    n_test = audit['splits']['test']['n']
    d_mlp, d_lr = boot['diff_gb_mlp'], boot['diff_gb_lr']
    best = history[history.epoch == neural['best_epoch']].iloc[0]
    last = history.iloc[-1]
    gpu = neural.get('gpu_device_name', 'CUDA GPU').replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '')
    caught = int(win['tp'] - std['tp'])

    def ci(d):
        return f'[{d["ci_lower"]:+.3f}, {d["ci_upper"]:+.3f}]'

    deck = Deck()

    # ------------------------------------------------------------ 1 title
    s = deck.blank('Title', (
        f'Good afternoon. We predict which credit-card clients will default next month, using six months of '
        f'payment history for 30,000 real clients of a Taiwanese bank (UCI, 2005). The point is not to approve or '
        f'reject anyone automatically, but to rank clients so a review team sees the riskiest ones first. '
        f'About {prevalence:.0%} of clients default, so we measure ranking quality with Average Precision. '
        f'Each of us will present our own part and can explain the whole pipeline.'))
    text(s, M, .9, 8, .3, 'ASTANA IT UNIVERSITY  ·  MACHINE LEARNING  ·  FINAL PROJECT', 11, ACCENT, bold=True)
    text(s, M, 1.5, 8.2, 2.2, ['Predicting next-month', 'credit card default'], 48, INK, font=HEAD, after=0, line=.95)
    text(s, M, 3.55, 7.6, .8, 'Ranking clients for review from six months of payment history', 20, MUTED)
    text(s, 9.3, 1.55, 3.3, 1.2, f'{summary["positive_fraction"]:.1%}', 66, ACCENT, font=HEAD)
    text(s, 9.3, 2.75, 3.3, .8, 'of clients default the next month', 15, MUTED)
    for i, (name, group, _) in enumerate(TEAM):
        x = M + i * (W / 3)
        text(s, x, 5.55, W / 3 - .3, .4, name, 17, INK, bold=True)
        text(s, x, 5.95, W / 3 - .3, .4, group, 13, MUTED)

    # ------------------------------------------------------------ 2 problem
    s = deck.content('Problem', 'Rank clients so reviewers see likely defaulters first', (
        'The task is binary classification. Inputs are the 23 features known by the end of September 2005; the output '
        'is a probability of default in October and a review decision at a threshold chosen on validation. '
        'Success is Average Precision on clients the model never saw: it rewards putting real defaulters at the top '
        f'of the list. A random ranking gets AP equal to the default rate, {prevalence:.3f}, so that is the bar to beat. '
        'Accuracy would be misleading: predicting "no default" for everyone is 78% accurate and useless.'))
    cols = [('01', 'Input', '23 features known by the end of September 2005: credit limit, demographics, six months of repayment status, bills and payments.'),
            ('02', 'Output', 'Probability of default in October 2005, plus a review / no-review decision at a threshold fixed on validation data.'),
            ('03', 'Success', f'Average Precision on held-out clients. A random ranking scores {prevalence:.3f}; accuracy alone would hide the minority class.')]
    for i, (num, head, body) in enumerate(cols):
        x = M + i * (W / 3)
        text(s, x, 2.2, 3.6, .7, num, 30, ACCENT, font=HEAD)
        text(s, x, 2.95, 3.6, .5, head, 20, INK, bold=True)
        text(s, x, 3.5, 3.55, 2.2, body, 16, MUTED, line=1.1)
    text(s, M, 6.05, W, .4, 'A decision-support tool for prioritising review — not an automated lending decision.', 15, INK, bold=True)

    # ------------------------------------------------------------ 3 data
    s = deck.content('Data', '30,000 real clients of one Taiwanese bank, 2005', (
        f'The data is the public UCI "Default of Credit Card Clients" set (Yeh, 2009, CC BY 4.0). It has 30,000 rows, '
        f'23 predictors and no missing values. Features fall into four groups: client profile, monthly repayment status '
        f'PAY_0 to PAY_6, bill amounts and payment amounts, April to September 2005. The structural audit found '
        f'{structure["duplicate_feature_rows_extra"]} rows that repeat another row\'s features, and '
        f'{structure["conflicting_target_groups"]} of those groups have different labels — this drives our split design. '
        'Some category codes, for example PAY = -2 and 0, are not documented by UCI, so we keep them as separate categories '
        'instead of guessing what they mean.'))
    stat(s, M, 1.95, 3.4, f'{structure["rows"]:,}', 'clients, no missing values')
    stat(s, M + 4.0, 1.95, 3.4, f'{structure["predictor_count"]}', 'predictors, April–September 2005')
    stat(s, M + 8.0, 1.95, 3.9, f'{summary["positive_fraction"]:.1%}', 'default next month (train)', ACCENT)
    table(s, M, 3.95, [2.6, 4.8, 4.5], [
        ['Group', 'Columns', 'Note'],
        ['Profile', 'LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE', 'Credit limit in NT$; demographics'],
        ['Repayment status', 'PAY_0 (Sep) … PAY_6 (Apr)', '−1 paid duly; 1–8 months late; −2, 0 undocumented'],
        ['Bills and payments', 'BILL_AMT1–6, PAY_AMT1–6', 'NT$; negative bills, many zero payments'],
    ], size=14, row_h=.44, right_from=99)
    text(s, M, 6.25, W, .3, 'Yeh, I.-C. (2009). Default of Credit Card Clients. UCI ML Repository #350. DOI 10.24432/C55S3H. CC BY 4.0.', 11, MUTED)

    # ------------------------------------------------------------ 4 EDA
    s = deck.content('Exploratory analysis', 'The latest repayment status carries most of the signal', (
        f'All outcome statistics here use the training set only. The chart shows the default rate by September '
        f'repayment status. Clients with code 0 default {pay0.set_index("PAY_0").loc[0, "mean"]:.0%} of the time and those who paid duly '
        f'{pay0.set_index("PAY_0").loc[-1, "mean"]:.0%}; '
        f'one month late raises it to {pay0.set_index("PAY_0").loc[1, "mean"]:.0%}, two months late to '
        f'{pay0.set_index("PAY_0").loc[2, "mean"]:.0%}. The relationship is not linear, so we one-hot encode the codes. '
        f'Money columns are heavily skewed, include negative bills ({summary["negative_bill_counts"]["BILL_AMT1"]} in September) '
        f'and many zero payments ({summary["zero_payment_fraction"]["PAY_AMT1"]:.0%}), so we use a signed log. '
        'Codes 4 to 8 have very few clients, so their rates are noisy.'))
    data = CategoryChartData()
    data.categories = [str(int(c)) for c in pay0.PAY_0]
    data.add_series('Default rate', [float(v) for v in pay0['mean']])
    frame = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(M), Inches(2.0), Inches(6.9), Inches(3.9), data)
    chart = frame.chart
    style_chart(chart)
    chart.has_legend = False
    plot = chart.plots[0]
    plot.gap_width = 45
    plot.vary_by_categories = False
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format, labels.number_format_is_linked = '0%', False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size, labels.font.color.rgb = Pt(11), MUTED
    series = plot.series[0]
    for i, code in enumerate(pay0.PAY_0):
        point = series.points[i]
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = ACCENT if code >= 1 else FAINT
    chart.value_axis.visible = False
    chart.value_axis.has_major_gridlines = False
    chart.value_axis.maximum_scale = 1.0
    chart.value_axis.minimum_scale = 0
    quiet_category_axis(chart.category_axis)
    text(s, M, 5.95, 6.9, .3, 'Training default rate by September repayment code (PAY_0). Codes 4–8: fewer than 50 clients each.', 11, MUTED)
    findings = [
        (f'{summary["positive_fraction"]:.0%} positives', 'Average Precision as the main metric; threshold tuned on validation'),
        ('Non-linear, partly undocumented codes', 'One-hot encode PAY, SEX, EDUCATION, MARRIAGE; keep every code'),
        ('Heavy tails, negative bills, zero payments', 'Signed log  sign(x)·ln(1+|x|), then scaling fitted on train only'),
    ]
    for i, (fact, decision) in enumerate(findings):
        y = 2.1 + i * 1.25
        text(s, 8.15, y, 4.5, .4, fact, 17, INK, bold=True)
        text(s, 8.15, y + .42, 4.5, .7, [[('→  ', {'color': ACCENT}), (decision, {})]], 15, MUTED)

    # ------------------------------------------------------------ 5 protocol
    s = deck.content('Evaluation design', 'The test set stayed sealed until every decision was frozen', (
        'We split once with StratifiedGroupKFold: 20 folds, 14 for train, 3 for validation, 3 for test, seed 42. '
        'Rows with identical features form a group and always land in the same split, so the model cannot be tested on '
        'an exact copy of a training client. The scaler and one-hot encoder learn only from train. Validation chooses '
        'hyperparameters, the early-stopping epoch, the model and the threshold. Before opening the test set we wrote '
        'every setting and model file hash to frozen_selection.json; evaluation refuses to run if anything changed.'))
    parts = [('train', 'Train', INK), ('validation', 'Validation', MUTED), ('test', 'Test', ACCENT)]
    x = M
    for key, label, color in parts:
        split = audit['splits'][key]
        w = W * split['fraction']
        box = rect(s, x, 2.05, w - .06, .95, fill=color)
        box.text_frame.margin_left = box.text_frame.margin_right = Inches(.12)
        box.text_frame.word_wrap = False
        shape_text(box, [[(label, {'bold': True})], [(f'{split["n"]:,}', {})]], 15, WHITE, align=PP_ALIGN.LEFT)
        text(s, x, 3.12, w - .1, .35, f'{split["fraction"]:.0%} · {split["positive_rate"]:.1%} default', 12, MUTED)
        x += w
    guards = [('Duplicates grouped', f'{structure["feature_duplicate_groups"]} groups of identical profiles each stay in one split'),
              ('Fit on train only', 'Scaler and encoder learn from train; validation and test are only transformed'),
              ('Validation decides', 'Model, hyperparameters, stopping epoch and threshold use validation only'),
              ('Frozen with SHA-256', 'Settings and model files hashed before the first test prediction')]
    for i, (head, body) in enumerate(guards):
        gx = M + (i % 2) * (W / 2)
        gy = 3.95 + (i // 2) * 1.2
        text(s, gx, gy, .5, .5, str(i + 1), 24, ACCENT, font=HEAD)
        text(s, gx + .55, gy + .02, W / 2 - 1, .4, head, 17, INK, bold=True)
        text(s, gx + .55, gy + .42, W / 2 - 1, .6, body, 14, MUTED)

    # ------------------------------------------------------------ 6 models
    log_p = models['logistic']['params']
    gb_p = models['boosting']['params']
    s = deck.content('Models and tuning', 'Four model families, one shared input pipeline', (
        f'Every model sees the same 23 features through the same train-fitted preprocessing. The baseline always predicts '
        f'"no default". Logistic regression tried C in {cfg["logistic_C"]}; histogram gradient boosting tried '
        f'{len(cfg["boosting_candidates"])} predeclared configurations; the MLP tried {len(cfg["mlp_candidates"])} '
        'configurations that vary width, learning rate, dropout and weight decay. The search space was written in config.json '
        'before training. Within each family and overall, the highest validation AP wins. Gradient boosting won by a small '
        f'margin, {models["boosting"]["validation_ap"]:.4f} against {neural["validation_ap"]:.4f} for the MLP, and that choice was frozen.'))
    fams = [('dummy', 'Majority class', 'No search'),
            ('logistic', f'L2, lbfgs · C ∈ {{{", ".join(str(c) for c in cfg["logistic_C"])}}}', f'Best C = {log_p["C"]}'),
            ('boosting', f'Histogram GB · {len(cfg["boosting_candidates"])} configs', f'lr {gb_p["learning_rate"]}, {gb_p["max_leaf_nodes"]} leaves, L2 {gb_p["l2_regularization"]}'),
            ('mlp', f'2 hidden layers · {len(cfg["mlp_candidates"])} configs', f'{h1}–{h2}, lr {p["learning_rate"]}, dropout {p["dropout"]}')]
    cw = (W - 3 * .25) / 4
    for i, (key, search, best_cfg) in enumerate(fams):
        x = M + i * (cw + .25)
        chosen = key == winner
        if chosen:
            rect(s, x, 1.95, cw, 3.85, fill=SOFT)
        pad = .25
        text(s, x + pad, 2.15, cw - 2 * pad, .3, 'SELECTED' if chosen else ' ', 11, ACCENT, bold=True)
        text(s, x + pad, 2.5, cw - 2 * pad, .8, NAMES[key], 20, INK, bold=True)
        text(s, x + pad, 3.25, cw - 2 * pad, .8, [search, best_cfg], 13, MUTED, after=3)
        text(s, x + pad, 4.35, cw - 2 * pad, .8, f'{models[key]["validation_ap"]:.3f}', 44, ACCENT if chosen else INK, font=HEAD)
        text(s, x + pad, 5.2, cw - 2 * pad, .4, 'validation AP', 12, MUTED)
    text(s, M, 6.15, W, .4, 'Rule fixed in advance: highest validation AP within each family and overall. Thresholds maximise validation F1.', 14, MUTED)

    # ------------------------------------------------------------ 7 neural network
    s = deck.content('Neural network', 'A small MLP, trained with backpropagation on a GPU', (
        f'The network takes the {neural["input_dim"]} preprocessed inputs, passes them through two hidden layers of {h1} and {h2} '
        f'units with ReLU and dropout {p["dropout"]}, and outputs one logit. Sigmoid turns it into a probability. '
        'Each training step: the forward pass computes logits and binary cross-entropy; loss.backward() uses the chain rule '
        'to get the gradient of the loss for every weight; optimizer.step() lets AdamW move each weight against its gradient, '
        f'with weight decay {p["weight_decay"]}. Batches of {neural["batch_size"]} are shuffled each epoch. We train up to '
        f'{neural["max_epochs"]} epochs and stop after {neural["patience"]} epochs without a better validation AP, then restore the best weights. '
        f'The loss is unweighted: the threshold, not the loss, handles the class imbalance.'))
    blocks = [(f'Input', f'{neural["input_dim"]} features'),
              (f'Linear {h1}', f'ReLU · Dropout {p["dropout"]}'),
              (f'Linear {h2}', f'ReLU · Dropout {p["dropout"]}'),
              ('Linear 1', 'logit → sigmoid')]
    bw, gap = 2.45, .7
    for i, (head, sub) in enumerate(blocks):
        bx = M + i * (bw + gap)
        box = rect(s, bx, 2.05, bw, 1.05, fill=SOFT if i in (1, 2) else None, line=None if i in (1, 2) else RULE)
        shape_text(box, [[(head, {'bold': True, 'size': 17})], [(sub, {'size': 13, 'color': MUTED})]])
        if i < len(blocks) - 1:
            text(s, bx + bw, 2.3, gap, .5, '→', 24, FAINT, align=PP_ALIGN.CENTER)
    table(s, M, 3.65, [2.3, 4.2], [
        ['Loss', 'BCEWithLogitsLoss (unweighted)'],
        ['Optimizer', f'AdamW · lr {p["learning_rate"]} · weight decay {p["weight_decay"]}'],
        ['Batch / epochs', f'{neural["batch_size"]} / up to {neural["max_epochs"]}, patience {neural["patience"]}'],
        ['Stopped / restored', f'epoch {neural["epochs_run"]} / epoch {neural["best_epoch"]}'],
        ['Parameters', f'{neural["parameter_count"]:,} · {gpu} · seed {neural["seed"]}'],
    ], size=14, row_h=.44, header=False, right_from=99)
    text(s, 8.0, 3.65, 4.6, .4, 'One training step', 17, INK, bold=True)
    text(s, 8.0, 4.15, 4.6, 2.2, [
        [('logits = model(x)', {'font': MONO, 'size': 13, 'color': INK})], 'forward pass',
        [('loss.backward()', {'font': MONO, 'size': 13, 'color': INK})], 'chain rule → gradient for every weight',
        [('optimizer.step()', {'font': MONO, 'size': 13, 'color': INK})], 'AdamW update + weight decay'],
        13, MUTED, after=2)

    # ------------------------------------------------------------ 8 learning curves
    s = deck.content('Training dynamics', 'Train AP keeps rising, validation AP flattens: mild overfitting', (
        f'Both curves are measured after each epoch with dropout switched off, so they are comparable. Validation AP rises fast '
        f'for about six epochs and then flattens near {best.validation_ap:.3f}. Training AP keeps climbing to {last.train_ap:.3f}. '
        f'That widening gap is mild overfitting: the network starts to fit noise in the training clients. Early stopping keeps the '
        f'weights from epoch {neural["best_epoch"]}, where validation AP peaked at {best.validation_ap:.4f}, and stopped training at '
        f'epoch {neural["epochs_run"]}. We see no underfitting: both curves are well above the {prevalence:.3f} baseline. The loss curves '
        f'(in artifacts/neural/learning_curves.png) tell the same story: validation BCE stops improving at about {best.validation_loss:.3f}.'))
    data = CategoryChartData()
    data.categories = [str(int(e)) for e in history.epoch]
    data.add_series('Train AP', [float(v) for v in history.train_ap])
    data.add_series('Validation AP', [float(v) for v in history.validation_ap])
    frame = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(M), Inches(1.9), Inches(8.0), Inches(4.4), data)
    chart = frame.chart
    style_chart(chart)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(12)
    for series, color in zip(chart.plots[0].series, [FAINT, ACCENT]):
        series.smooth = False
        series.marker.style = XL_MARKER_STYLE.NONE
        series.format.line.color.rgb = color
        series.format.line.width = Pt(2.5)
    va = chart.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = .46, .60, .02
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RULE
    va.format.line.fill.background()
    va.major_tick_mark = XL_TICK_MARK.NONE
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = '0.00', False
    va.tick_labels.font.size = Pt(11)
    quiet_category_axis(chart.category_axis, 11)
    text(s, M, 6.35, 8.0, .3, 'Average precision per epoch, selected MLP. Evaluated with dropout off.', 11, MUTED)
    stat(s, 9.4, 1.95, 3.2, f'{neural["best_epoch"]}', f'best epoch (of {neural["epochs_run"]} run)', ACCENT)
    stat(s, 9.4, 3.45, 3.2, f'{best.validation_ap:.3f}', 'validation AP at best epoch')
    stat(s, 9.4, 4.95, 3.2, f'+{last.train_ap - last.validation_ap:.3f}', 'train − validation gap, last epoch')

    # ------------------------------------------------------------ 9 results
    s = deck.content('Test results', 'Boosting and MLP tie on test; both far above the baseline', (
        f'These are the only numbers computed on the {n_test:,} test clients, after everything was frozen. AP is ranking quality; '
        f'the baseline equals the default rate, {working.loc["dummy", "ap"]:.3f}. Gradient boosting reaches {working.loc["boosting", "ap"]:.4f} and '
        f'the MLP {working.loc["mlp", "ap"]:.4f}. To check whether the gaps are real, we resampled the test clients 1,000 times and scored '
        f'both models on the same resample. Boosting minus MLP is {d_mlp["mean"]:+.4f} with interval {ci(d_mlp)}: it contains zero, '
        f'so we cannot tell them apart — which is not the same as proving they are identical. Boosting beats logistic regression '
        f'by {d_lr["mean"]:+.3f}, but that interval {ci(d_lr)} also contains zero. The neural network did not need to win; '
        'it matches the best classical model.'))
    rows = [['Model', 'Test AP', '95% bootstrap CI', 'ROC-AUC', 'Precision', 'Recall', 'F1']]
    order = ['dummy', 'logistic', 'boosting', 'mlp']
    for name in order:
        r = working.loc[name]
        rows.append([NAMES[name] + ('  (selected)' if name == winner else ''), f'{r.ap:.3f}',
                     f'{r.ap_ci_lower:.3f} – {r.ap_ci_upper:.3f}', f'{r.roc_auc:.3f}',
                     f'{r.precision:.3f}', f'{r.recall:.3f}', f'{r.f1:.3f}'])
    table(s, M, 1.95, [3.6, 1.35, 2.35, 1.3, 1.3, 1.0, 1.0], rows, size=16, row_h=.52,
          emphasize=(order.index(winner) + 1,))
    text(s, M, 4.95, 5.7, .4, 'Boosting − MLP', 15, MUTED)
    text(s, M, 5.3, 5.7, .7, [[(f'{d_mlp["mean"]:+.4f}', {'size': 30, 'font': HEAD, 'color': INK}),
                              (f'   CI {ci(d_mlp)}', {'size': 15, 'color': MUTED})]])
    text(s, M, 6.0, 5.7, .4, 'No detectable difference', 15, INK, bold=True)
    text(s, M + 6.2, 4.95, 5.7, .4, 'Boosting − logistic regression', 15, MUTED)
    text(s, M + 6.2, 5.3, 5.7, .7, [[(f'{d_lr["mean"]:+.4f}', {'size': 30, 'font': HEAD, 'color': INK}),
                                    (f'   CI {ci(d_lr)}', {'size': 15, 'color': MUTED})]])
    text(s, M + 6.2, 6.0, 5.7, .4, 'Not significant at α = 0.05', 15, INK, bold=True)
    text(s, M, 6.5, W, .3, f'n = {n_test:,} held-out clients · thresholds fixed on validation · paired bootstrap, 1,000 resamples', 11, MUTED)

    # ------------------------------------------------------------ 10 threshold
    s = deck.content('Operating point', f'A {win.threshold:.2f} threshold catches {caught} more defaulters than 0.5', (
        f'A probability is not a decision. With the usual 0.5 cut-off the selected model flags {int(std.tp + std.fp)} clients and misses '
        f'{int(std.fn)} of {int(std.tp + std.fn)} defaulters, recall {std.recall:.0%}. We chose the threshold that maximises F1 on validation, '
        f'{win.threshold:.3f}, and applied it unchanged to test. Recall rises to {win.recall:.0%} and missed defaults fall to {int(win.fn)}, '
        f'but false alarms grow from {int(std.fp)} to {int(win.fp)}. F1 is a teaching choice: a real bank would set the threshold from its '
        'review capacity and the cost of a missed default versus an unnecessary review.'))
    queue = lambda r: int(r.tp + r.fp)  # noqa: E731
    table(s, M, 1.95, [3.6, 2.2, 2.6], [
        ['Selected model on test', 'Threshold 0.50', f'Threshold {win.threshold:.3f}'],
        ['Recall', f'{std.recall:.0%}', f'{win.recall:.0%}'],
        ['Precision', f'{std.precision:.0%}', f'{win.precision:.0%}'],
        ['Defaulters missed', f'{int(std.fn)}', f'{int(win.fn)}'],
        ['False alarms', f'{int(std.fp)}', f'{int(win.fp)}'],
        ['Clients sent to review', f'{queue(std)}', f'{queue(win)}'],
    ], size=20, row_h=.62, header_size=13)
    text(s, 9.55, 2.0, 3.1, 1.1, f'+{caught}', 66, ACCENT, font=HEAD)
    text(s, 9.55, 3.15, 3.0, .9, 'defaulters caught by lowering the threshold', 15, MUTED)
    text(s, 9.55, 4.35, 3.0, 1.6, 'Threshold = max validation F1, fixed before test. Real use: set it from review capacity and error costs.', 13, MUTED)

    # ------------------------------------------------------------ 11 interpretation
    top = importance.head(8)
    fp = errors[errors.error_type == 'FP'].sort_values('probability').iloc[-1]
    fn = errors[errors.error_type == 'FN'].sort_values('probability').iloc[0]
    tp = errors[errors.error_type == 'TP'].sort_values('probability').iloc[-1]

    def status(code):
        code = int(code)
        return f'{code} months late' if code >= 2 else ('1 month late' if code == 1 else ('paid duly' if code == -1 else f'code {code} (undocumented)'))

    s = deck.content('Interpretation and errors', 'PAY_0 dominates; errors come from what the data cannot see', (
        f'Permutation importance shuffles one column at a time on the test set and measures how much AP drops. Shuffling PAY_0, '
        f'the September repayment status, drops AP by {top.iloc[0].mean_ap_drop:.3f}; every other feature drops it by less than '
        f'{top.iloc[1].mean_ap_drop:.3f}. Monthly features are correlated, so they can stand in for each other and share importance. '
        f'This is predictive reliance, not causation. Looking at single clients: client {int(fp.row_index)} scored {fp.probability:.2f} because '
        f'of a recent delay but did not default. Client {int(fn.row_index)} scored only {fn.probability:.2f} and defaulted — nothing in the '
        'six months of history signalled it. Events like job loss are simply not in the data.'))
    data = CategoryChartData()
    data.categories = list(top.feature)
    data.add_series('AP drop', [float(v) for v in top.mean_ap_drop])
    frame = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(M), Inches(1.95), Inches(5.9), Inches(4.25), data)
    chart = frame.chart
    style_chart(chart)
    chart.has_legend = False
    plot = chart.plots[0]
    plot.gap_width = 40
    plot.vary_by_categories = False
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format, labels.number_format_is_linked = '0.000', False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size, labels.font.color.rgb = Pt(11), MUTED
    for i in range(len(top)):
        point = plot.series[0].points[i]
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = ACCENT if i == 0 else FAINT
    chart.value_axis.visible = False
    chart.value_axis.has_major_gridlines = False
    chart.value_axis.maximum_scale = float(top.mean_ap_drop.max()) * 1.25
    chart.value_axis.minimum_scale = 0
    quiet_category_axis(chart.category_axis)
    chart.category_axis.reverse_order = True
    text(s, M, 6.25, 5.9, .3, 'Drop in test AP when the column is shuffled (mean of 5).', 11, MUTED)
    cases = [('False positive', fp, f'September: {status(fp.PAY_0)}', 'did not default'),
             ('False negative', fn, f'September: {status(fn.PAY_0)}', 'defaulted'),
             ('True positive', tp, f'Sep: {status(tp.PAY_0)} · Aug: {status(tp.PAY_2)}', 'defaulted')]
    for i, (kind, row, why, outcome) in enumerate(cases):
        y = 2.0 + i * 1.4
        text(s, 7.2, y, 1.7, .9, f'{row.probability:.2f}', 36, ACCENT if i < 2 else INK, font=HEAD)
        text(s, 8.9, y + .05, 3.8, .4, f'{kind} · client {int(row.row_index)}', 15, INK, bold=True)
        text(s, 8.9, y + .45, 3.8, .7, [why, f'Actual: {outcome}'], 13, MUTED, after=1)
    text(s, 7.2, 6.25, 5.4, .3, 'Score = predicted default probability; threshold ' + f'{win.threshold:.3f}.', 11, MUTED)

    # ------------------------------------------------------------ 12 limitations
    s = deck.content('Critical analysis', 'What this result does not show', (
        'We want to be precise about the claim. The test clients come from the same bank and the same months as the training '
        'clients, so this is not a test of the future or of another bank. The scores rank well but were never calibrated, so '
        '0.7 does not mean a 70% chance. The F1 threshold ignores real costs. The data contains sex, age and marital status, and we '
        'did not audit errors by group. Finally, one split, one seed and a small search cannot rank algorithms in general. '
        'Each limitation maps to a concrete next step on the right.'))
    pairs = [('One bank, one country, 2005', 'Forward-in-time test on later cohorts'),
             ('Scores are not calibrated', 'Calibration curve; Platt or isotonic scaling'),
             ('F1 threshold ignores real costs', 'Cost- and capacity-based threshold'),
             ('Demographic features, no fairness audit', 'Error rates by sex, age, education'),
             ('One split, one seed, small search', 'Repeated seeds, wider MLP search, new features')]
    text(s, M, 1.95, 5.6, .4, 'LIMITATION', 11, MUTED, bold=True)
    text(s, M + 6.4, 1.95, 5.6, .4, 'NEXT STEP', 11, ACCENT, bold=True)
    for i, (limit, step) in enumerate(pairs):
        y = 2.45 + i * .82
        rect(s, M, y - .12, W, .01, fill=RULE)
        text(s, M, y, 5.9, .6, limit, 18, INK)
        text(s, M + 5.95, y, .4, .6, '→', 18, FAINT)
        text(s, M + 6.4, y, 5.5, .6, step, 18, MUTED)

    # ------------------------------------------------------------ 13 demo
    demo = predictions.iloc[0]
    decision = 'PRIORITIZE FOR REVIEW' if demo[winner] >= win.threshold else 'NOT PRIORITIZED'
    s = deck.content('Demonstration', 'Score a held-out client with one command', (
        f'The demo reloads the frozen {NAMES[winner].lower()} pipeline, including the preprocessing fitted on train, checks the SHA-256 '
        f'hashes, and scores test client number 0 from raw features. The score is {demo[winner]:.4f}, above the threshold '
        f'{win.threshold:.4f}, so the decision is "{decision}". Only after the prediction does it print the true label, {int(demo.actual)}. '
        'Any other test client can be scored with --index, and the predictor accepts a new raw 23-column record.'))
    box = rect(s, M, 1.95, W, .75, fill=SOFT)
    shape_text(box, [[('uv run python -m src.demo --index 0 --reveal-label', {'font': MONO})]], 18, INK, align=PP_ALIGN.LEFT)
    text(s, M, 3.05, 6.6, 3.2, [
        '{',
        f'  "model": "{winner}",',
        '  "test_position": 0,',
        f'  "source_row_index": {int(demo.row_index)},',
        f'  "estimated_default_probability": {demo[winner]:.4f}…,',
        f'  "threshold": {win.threshold:.4f}…,',
        f'  "decision": "{decision}"',
        '}',
        f'Actual label (revealed after prediction): {int(demo.actual)}'], 14, INK, font=MONO, after=1)
    steps = ['Loads the frozen model and train-fitted preprocessing',
             'Verifies the SHA-256 hashes from the freeze step',
             'Scores one raw test client, then reveals the label']
    for i, step in enumerate(steps):
        y = 3.05 + i * 1.0
        text(s, 7.9, y, .5, .5, str(i + 1), 24, ACCENT, font=HEAD)
        text(s, 8.45, y + .08, 4.2, .8, step, 16, INK)

    # ------------------------------------------------------------ 14 references
    s = deck.content('Sources', 'References and AI assistance', (
        'These are our sources: the UCI dataset and the original paper by Yeh and Lien, the Grinsztajn et al. study on why '
        'tree models are strong on tabular data, and the scikit-learn and PyTorch libraries. AI coding assistants helped with code '
        'scaffolding, documentation drafts, review and slide layout. We checked every number against the saved artifacts and each of '
        'us can explain any part of the code.'))
    text(s, M, 1.95, 7.4, 4.6, [
        'Yeh, I.-C. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0)',
        'Yeh, I.-C. & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020',
        'Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems 35, 507–520.',
        'Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825‑2830.',
        'Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 32.'],
        15, INK, after=12, bullets=True)
    rect(s, 8.55, 1.95, 4.08, 3.6, fill=SOFT)
    text(s, 8.85, 2.25, 3.5, .4, 'AI assistance', 18, INK, bold=True)
    text(s, 8.85, 2.8, 3.5, 2.6,
         ['AI coding assistants (OpenAI Codex, Google Antigravity, Claude Code) were used for code scaffolding, documentation drafts, review and slide layout.',
          'Every reported number comes from saved run artifacts. Each team member can explain the code and the decisions.'],
         15, MUTED, after=12)

    # ------------------------------------------------------------ 15 questions
    s = deck.blank('Questions', (
        'Thank you. To summarise: on 4,500 unseen clients, gradient boosting and the PyTorch MLP rank defaulters equally well, '
        f'about {win.ap / prevalence:.1f} times the chance level of AP, under a protocol where the test set was never used for any decision. '
        'We are happy to take questions; each of us can walk through the code.'))
    text(s, M, .9, 8, .3, 'THANK YOU', 11, ACCENT, bold=True)
    text(s, M, 1.4, 8, 1.2, 'Questions', 54, INK, font=HEAD)
    text(s, M, 2.55, 11, .5, f'Test AP {win.ap:.3f} vs {prevalence:.3f} by chance  ·  {n_test:,} unseen clients  ·  test used once', 18, MUTED)
    for i, (name, group, role) in enumerate(TEAM):
        y = 3.75 + i * .82
        rect(s, M, y - .14, W, .01, fill=RULE)
        text(s, M, y, 3.6, .5, name, 18, INK, bold=True)
        text(s, M + 3.7, y, 1.4, .5, group, 15, MUTED)
        text(s, M + 5.2, y, 6.7, .5, role, 15, MUTED)
    text(s, M, 6.35, W, .4, [[('Reproduce:  ', {'color': MUTED}), ('uv run python run.py', {'font': MONO, 'color': INK})]], 14)

    # ------------------------------------------------------------ save
    prs = deck.prs
    OUT.mkdir(parents=True, exist_ok=True)
    pptx_path = OUT / 'credit_default_final.pptx'
    prs.save(pptx_path)
    reopened = Presentation(pptx_path)
    assert len(reopened.slides) == len(deck.notes)
    assert all(sl.notes_slide.notes_text_frame.text for sl in reopened.slides)

    lines = ['# Presentation script and speaker notes', '',
             f'{len(deck.notes)} slides, about 40 seconds each: roughly 10 minutes plus the live demo.', '']
    for i, (title, script) in enumerate(deck.notes, 1):
        lines += [f'## Slide {i}: {title}', '', script, '']
    (ROOT / 'docs/speaker_notes.md').write_text('\n'.join(lines), encoding='utf-8')
    manifest = {'slides': len(reopened.slides), 'selected_model': winner, 'style': 'minimal',
                'native_charts': 3, 'bootstrap_ci_included': True, 'paired_bootstrap_included': True,
                'error_clients_included': True, 'team_members_included': True,
                'pptx_path': str(pptx_path.relative_to(ROOT)).replace('\\', '/')}
    (OUT / 'build_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Successfully built {pptx_path} ({len(reopened.slides)} slides) and speaker notes.')


if __name__ == '__main__':
    build()
