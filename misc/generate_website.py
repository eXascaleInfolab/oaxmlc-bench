#!/usr/bin/env python3
"""
Generate a single static HTML page (docs/index.html) containing:
- Intro / datasets / metrics (with MathJax-rendered formulas) / tasks / methods / references
- All CSV tables found in --in_dir, grouped by dataset then experiment
- Click-to-sort columns (numeric-aware)
- Prettified UI (sticky nav, hero header, zebra tables, sticky first column, back-to-top, theme toggle)

Usage:
  python make_tables_site.py --in_dir path/to/csvs --out docs/index.html
"""

from __future__ import annotations

from pathlib import Path
import argparse
import html
import pandas as pd
import re


# -----------------------------
# Pretty names
# -----------------------------

ALG_PRETTY = {
    "match": "MATCH",
    "xmlcnn": "XML-CNN",
    "xml-cnn": "XML-CNN",
    "attentionxml": "AttentionXML",
    "fastxml": "FastXML",
    "hector": "Hector",
    "tamlec": "TAMLeC",
    "lightxml": "LightXML",
    "cascadexml": "CascadeXML",
    "parabel": "Parabel",
    "ngame": "NGAME",
    "dexa": "DEXA",
}

# NOTE: these keys match your CSV headers exactly.
METRIC_PRETTY = {
    # ---- base metrics ----
    "P@1": "P@1",
    "P@2": "P@2",
    "P@3": "P@3",
    "RS@1": "R@1",
    "RS@2": "R@2",
    "N@3": "nDCG@3",
    "MaP": "Macro-P",
    "MaR": "Macro-R",
    "MaF1": "Macro-F1",
    r"$\mu$P": "Micro-P",
    r"$\mu$R": "Micro-R",
    r"$\mu$F1": "Micro-F1",

    # ---- per-level (L1/L2/L3) ----
    "P@1_L1": "P@1 (L1)",
    "P@2_L1": "P@2 (L1)",
    "P@3_L1": "P@3 (L1)",
    "RS@1_L1": "R@1 (L1)",
    "RS@2_L1": "R@2 (L1)",
    "N@3_L1": "nDCG@3 (L1)",
    r"$\mu$P_L1": "Micro-P (L1)",
    r"$\mu$R_L1": "Micro-R (L1)",
    r"$\mu$F1_L1": "Micro-F1 (L1)",
    "MaP_L1": "Macro-P (L1)",
    "MaR_L1": "Macro-R (L1)",
    "MaF1_L1": "Macro-F1 (L1)",

    "P@1_L2": "P@1 (L2)",
    "P@2_L2": "P@2 (L2)",
    "P@3_L2": "P@3 (L2)",
    "RS@1_L2": "R@1 (L2)",
    "RS@2_L2": "R@2 (L2)",
    "N@3_L2": "nDCG@3 (L2)",
    r"$\mu$P_L2": "Micro-P (L2)",
    r"$\mu$R_L2": "Micro-R (L2)",
    r"$\mu$F1_L2": "Micro-F1 (L2)",
    "MaP_L2": "Macro-P (L2)",
    "MaR_L2": "Macro-R (L2)",
    "MaF1_L2": "Macro-F1 (L2)",

    "P@1_L3": "P@1 (L3)",
    "P@2_L3": "P@2 (L3)",
    "P@3_L3": "P@3 (L3)",
    "RS@1_L3": "R@1 (L3)",
    "RS@2_L3": "R@2 (L3)",
    "N@3_L3": "nDCG@3 (L3)",
    r"$\mu$P_L3": "Micro-P (L3)",
    r"$\mu$R_L3": "Micro-R (L3)",
    r"$\mu$F1_L3": "Micro-F1 (L3)",
    "MaP_L3": "Macro-P (L3)",
    "MaR_L3": "Macro-R (L3)",
    "MaF1_L3": "Macro-F1 (L3)",

    # ---- Few-shot FT vs NoFT ----
    "P@1_NoFT": "P@1 (no FT)",
    "P@2_NoFT": "P@2 (no FT)",
    "P@3_NoFT": "P@3 (no FT)",
    "RS@1_NoFT": "R@1 (no FT)",
    "RS@2_NoFT": "R@2 (no FT)",
    "N@3_NoFT": "nDCG@3 (no FT)",
    r"$\mu$P_NoFT": "Micro-P (no FT)",
    r"$\mu$R_NoFT": "Micro-R (no FT)",
    r"$\mu$F1_NoFT": "Micro-F1 (no FT)",
    "MaP_NoFT": "Macro-P (no FT)",
    "MaR_NoFT": "Macro-R (no FT)",
    "MaF1_NoFT": "Macro-F1 (no FT)",

    "P@1_FT": "P@1 (FT)",
    "P@2_FT": "P@2 (FT)",
    "P@3_FT": "P@3 (FT)",
    "RS@1_FT": "R@1 (FT)",
    "RS@2_FT": "R@2 (FT)",
    "N@3_FT": "nDCG@3 (FT)",
    r"$\mu$P_FT": "Micro-P (FT)",
    r"$\mu$R_FT": "Micro-R (FT)",
    r"$\mu$F1_FT": "Micro-F1 (FT)",
    "MaP_FT": "Macro-P (FT)",
    "MaR_FT": "Macro-R (FT)",
    "MaF1_FT": "Macro-F1 (FT)",
}

TABLE_TITLES = {
    "table_classification_level_oamedconcepts.csv": "Classification (per level) — OAXMLC-Med Concepts",
    "table_classification_level_oamedtopics.csv": "Classification (per level) — OAXMLC-Med Topics",
    "table_classification_level_oaxmlc_concepts.csv": "Classification (per level) — OAXMLC-CS Concepts",
    "table_classification_level_oaxmlc_topics.csv": "Classification (per level) — OAXMLC-CS Topics",
    "table_classification_oamedconcepts.csv": "Classification — OAXMLC-Med Concepts",
    "table_classification_oamedtopics.csv": "Classification — OAXMLC-Med Topics",
    "table_classification_oaxmlc_concepts.csv": "Classification — OAXMLC-CS Concepts",
    "table_classification_oaxmlc_topics.csv": "Classification — OAXMLC-CS Topics",
    "table_completion_oamedconcepts.csv": "Completion — OAXMLC-Med Concepts",
    "table_completion_oamedtopics.csv": "Completion — OAXMLC-Med Topics",
    "table_completion_oaxmlc_concepts.csv": "Completion — OAXMLC-CS Concepts",
    "table_completion_oaxmlc_topics.csv": "Completion — OAXMLC-CS Topics",
    "table_fewshot_global_oamedconcepts.csv": "Few-shot (global) — OAXMLC-Med Concepts",
    "table_fewshot_global_oamedtopics.csv": "Few-shot (global) — OAXMLC-Med Topics",
    "table_fewshot_global_oaxmlc_concepts.csv": "Few-shot (global) — OAXMLC-CS Concepts",
    "table_fewshot_global_oaxmlc_topics.csv": "Few-shot (global) — OAXMLC-CS Topics",
    "table_fewshot_task_oamedconcepts.csv": "Few-shot (task) — OAXMLC-Med Concepts",
    "table_fewshot_task_oamedtopics.csv": "Few-shot (task) — OAXMLC-Med Topics",
    "table_fewshot_task_oaxmlc_concepts.csv": "Few-shot (task) — OAXMLC-CS Concepts",
    "table_fewshot_task_oaxmlc_topics.csv": "Few-shot (task) — OAXMLC-CS Topics",
}

DATASET_PRETTY = {
    "oaxmlc_concepts": "OAXMLC-CS Concepts",
    "oaxmlc_topics": "OAXMLC-CS Topics",
    "oamedconcepts": "OAXMLC-Med Concepts",
    "oamedtopics": "OAXMLC-Med Topics",
}

DATASET_ALIASES = {
    "oaxmlc_concepts": "oaxmlc_concepts",
    "oaxmlcconcepts": "oaxmlc_concepts",
    "oaxmlc_topics": "oaxmlc_topics",
    "oaxmlctopics": "oaxmlc_topics",
    "oamedconcepts": "oamedconcepts",
    "oamedtopics": "oamedtopics",
}

DATASET_ORDER = [
    "oaxmlc_concepts",
    "oaxmlc_topics",
    "oamedconcepts",
    "oamedtopics",
]

EXPERIMENT_PRETTY = {
    "classification": "Classification",
    "classification_level": "Classification (per level)",
    "classification_levels": "Classification (per level)",
    "completion": "Completion",
    "fewshot": "Few-shot",
    "fewshot_global": "Few-shot (global)",
    "fewshot_task": "Few-shot (task)",
}

EXPERIMENT_ORDER = [
    "classification",
    "classification_level",
    "classification_levels",
    "completion",
    "fewshot_global",
    "fewshot_task",
    "fewshot",
]


# -----------------------------
# Sections (HTML fragments)
# -----------------------------

METRICS_FORMULAS_HTML = r"""
<div class="section" id="metrics"><h2>Metrics</h2></div>
<div class="metrics">
  <p>
    We report both thresholded <b>classification</b> metrics (Micro/Macro P, R, F1) and
    <b>ranking</b> metrics (P@k, R@k, nDCG@k). For classification metrics, scores are binarized
    using a threshold selected on the validation set (e.g., maximizing Micro-F1).
  </p>

  <h4>Notation</h4>
  <p>
    Let \(N\) be the number of documents and \(L\) the number of labels.
    For document \(i\), the set of true labels is \(Y_i\), and the predicted ranking is
    \(\hat{\pi}_i=(\hat{\pi}_{i,1},\hat{\pi}_{i,2},\dots)\).
    After thresholding, \(\hat{y}_{i\ell}\in\{0,1\}\) denotes whether label \(\ell\) is predicted for document \(i\),
    and \(y_{i\ell}\in\{0,1\}\) is the ground-truth.
  </p>

  <div class="wrap" style="margin-top:10px;">
    <table>
      <thead>
        <tr><th>Metric</th><th>Definition</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><b>Micro-P / Micro-R / Micro-F1</b></td>
          <td style="white-space:normal;">
            \[
              \mathrm{TP}_\ell=\sum_{i=1}^N \mathbb{1}[\hat{y}_{i\ell}=1 \wedge y_{i\ell}=1],\quad
              \mathrm{FP}_\ell=\sum_{i=1}^N \mathbb{1}[\hat{y}_{i\ell}=1 \wedge y_{i\ell}=0],\quad
              \mathrm{FN}_\ell=\sum_{i=1}^N \mathbb{1}[\hat{y}_{i\ell}=0 \wedge y_{i\ell}=1]
            \]
            \[
              P_{\mu}=\frac{\sum_{\ell=1}^{L}\mathrm{TP}_{\ell}}{\sum_{\ell=1}^{L}(\mathrm{TP}_{\ell}+\mathrm{FP}_{\ell})},\quad
              R_{\mu}=\frac{\sum_{\ell=1}^{L}\mathrm{TP}_{\ell}}{\sum_{\ell=1}^{L}(\mathrm{TP}_{\ell}+\mathrm{FN}_{\ell})},\quad
              F1_{\mu}=\frac{2P_{\mu}R_{\mu}}{P_{\mu}+R_{\mu}}
            \]
          </td>
        </tr>

        <tr>
          <td><b>Macro-P / Macro-R / Macro-F1</b></td>
          <td style="white-space:normal;">
            \[
              P_{\ell}=\frac{\mathrm{TP}_{\ell}}{\mathrm{TP}_{\ell}+\mathrm{FP}_{\ell}},\quad
              R_{\ell}=\frac{\mathrm{TP}_{\ell}}{\mathrm{TP}_{\ell}+\mathrm{FN}_{\ell}},\quad
              F1_{\ell}=\frac{2P_{\ell}R_{\ell}}{P_{\ell}+R_{\ell}}
            \]
            \[
              P_{M}=\frac{1}{L}\sum_{\ell=1}^{L}P_{\ell},\quad
              R_{M}=\frac{1}{L}\sum_{\ell=1}^{L}R_{\ell},\quad
              F1_{M}=\frac{1}{L}\sum_{\ell=1}^{L}F1_{\ell}
            \]
          </td>
        </tr>

        <tr>
          <td><b>P@k</b></td>
          <td style="white-space:normal;">
            \[
              P@k=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{k}\sum_{j=1}^{k}\mathbb{1}[\hat{\pi}_{i,j}\in Y_i]
            \]
          </td>
        </tr>

        <tr>
          <td><b>R@k</b></td>
          <td style="white-space:normal;">
            \[
              R@k=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|Y_i|}\sum_{j=1}^{k}\mathbb{1}[\hat{\pi}_{i,j}\in Y_i]
            \]
          </td>
        </tr>

        <tr>
          <td><b>nDCG@k (N@k)</b></td>
          <td style="white-space:normal;">
            \[
              \mathrm{rel}_{i,j}=\mathbb{1}[\hat{\pi}_{i,j}\in Y_i],\quad
              \mathrm{DCG}@k(i)=\sum_{j=1}^{k}\frac{\mathrm{rel}_{i,j}}{\log_2(j+1)},\quad
              \mathrm{IDCG}@k(i)=\sum_{j=1}^{\min(k,|Y_i|)}\frac{1}{\log_2(j+1)}
            \]
            \[
              \mathrm{nDCG}@k=\frac{1}{N}\sum_{i=1}^{N}\frac{\mathrm{DCG}@k(i)}{\mathrm{IDCG}@k(i)}
            \]
          </td>
        </tr>
      </tbody>
    </table>
  </div>

  <p style="margin-top:10px;">
    Note: when computing ranking metrics at cutoff \(k\), you may restrict to documents with \(|Y_i|\ge k\).
    In that case, P@3 can be larger than P@2 because they are averaged over different subsets.
  </p>
</div>
"""

TASKS_HTML = r"""
<div class="section" id="tasks"><h2>Tasks</h2></div>
<div class="datasets">
  <p>
    The OAXMLC benchmark evaluates Extreme Multi-Label Classification (XMLC) methods across multiple tasks,
    each probing a distinct capability of taxonomy-aware and taxonomy-agnostic models.
    All tasks assume the <em>taxonomy completion hypothesis</em>: if a document is annotated with a label,
    it is implicitly associated with all of its ancestor labels in the taxonomy.
  </p>

  <h4>XML Classification (XMLCl)</h4>
  <p>
    The standard XMLC setup: predict the full set of relevant labels for unseen documents using only their text.
    Models are trained on fully annotated documents and evaluated on a held-out test set.
  </p>
  <ul>
    <li><b>Global classification</b>: performance across all labels in the taxonomy (excluding the root).</li>
    <li><b>Per-level classification</b>: performance evaluated separately at each taxonomy depth.</li>
  </ul>

  <h4>XML Completion (XMLCo)</h4>
  <p>
    Documents are <em>partially annotated</em>: each document is initially provided with general labels (e.g., level-1),
    and the goal is to predict missing, more specific labels at deeper levels.
    This reflects realistic settings with incomplete annotations or evolving taxonomies.
  </p>

  <h4>XML Few-Shot Learning (XMLFS)</h4>
  <p>
    Evaluate adaptation to <em>new labels</em>. During training, a whole subtree is withheld and its labels are removed.
    After convergence, models are exposed to a small number of labeled examples from the withheld subtree and fine-tuned.
    We report performance before and after fine-tuning.
  </p>
</div>
"""

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OAXMLC-Bench 𓊳</title>

  <style>
    :root {{
      --bg: #ffffff;
      --panel: #fafafa;
      --text: #111;
      --muted: #555;
      --border: #e6e6e6;
      --shadow: 0 10px 30px rgba(0,0,0,.06);
      --shadow2: 0 6px 18px rgba(0,0,0,.08);
      --accent: #3b82f6;
      --accent2: #0ea5e9;
    }}

    /* Manual theme override (set by JS) */
    html[data-theme="dark"] {{
      --bg: #0b0f17;
      --panel: #0f1623;
      --text: #e7eaf0;
      --muted: #a7b0c0;
      --border: #1f2a3a;
      --shadow: 0 10px 30px rgba(0,0,0,.45);
      --shadow2: 0 6px 18px rgba(0,0,0,.55);
      --accent: #60a5fa;
      --accent2: #38bdf8;
    }}
    html[data-theme="light"] {{
      /* keep :root values */
    }}

    /* If no manual override, follow system preference */
    @media (prefers-color-scheme: dark) {{
      html:not([data-theme]) {{
        --bg: #0b0f17;
        --panel: #0f1623;
        --text: #e7eaf0;
        --muted: #a7b0c0;
        --border: #1f2a3a;
        --shadow: 0 10px 30px rgba(0,0,0,.45);
        --shadow2: 0 6px 18px rgba(0,0,0,.55);
        --accent: #60a5fa;
        --accent2: #38bdf8;
      }}
    }}

    * {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial,
        "Apple Color Emoji","Segoe UI Emoji";
      margin: 0;
      color: var(--text);
      background: var(--bg);
    }}

    .container {{
      max-width: 1140px;
      margin: 0 auto;
      padding: 18px 18px 86px;
    }}

    /* Sticky top nav */
    .topnav {{
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(10px);
      background: color-mix(in srgb, var(--bg) 88%, transparent);
      border-bottom: 1px solid var(--border);
    }}
    .topnav-inner {{
      max-width: 1140px;
      margin: 0 auto;
      padding: 10px 18px;
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }}
    .navlinks {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 12px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }}
    .navlinks a {{
      text-decoration: none;
      color: inherit;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid transparent;
    }}
    .navlinks a:hover {{
      border-color: var(--border);
      background: var(--panel);
      color: var(--text);
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      cursor: pointer;
      user-select: none;
    }}
    .btn:hover {{ box-shadow: var(--shadow2); }}

    /* Hero */
    .hero {{
      margin-top: 14px;
      background: linear-gradient(120deg, rgba(59,130,246,.12), rgba(14,165,233,.07));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px 18px;
      box-shadow: var(--shadow);
    }}

    /* Headings */
    h1, h2, h3, h4 {{ text-align: left; margin: 0; }}
    h1 {{ font-size: 34px; letter-spacing: -0.02em; margin-bottom: 6px; }}
    h2.subtitle {{ font-size: 18px; color: var(--muted); font-weight: 600; margin-bottom: 12px; }}
    h3 {{ font-size: 22px; margin-top: 26px; margin-bottom: 10px; }}
    h4 {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .10em;
      color: var(--muted);
      margin-top: 18px;
      margin-bottom: 8px;
    }}

    .authors {{
      text-align: left;
      color: var(--muted);
      margin: 6px 0 0;
      line-height: 1.35;
    }}
    .authors em {{ color: var(--text); font-style: normal; font-weight: 700; }}

    /* Text blocks */
    .intro, .datasets, .metrics, .methods, .ref {{
      max-width: 920px;
      line-height: 1.6;
      color: var(--text);
    }}
    .intro p, .datasets p, .metrics p, .ref p {{ margin: 10px 0; }}
    .intro ul, .datasets ul, .metrics ul {{ margin: 10px 0 0 18px; padding: 0; }}
    .intro li, .datasets li, .metrics li {{ margin: 6px 0; }}

    .toc {{
      max-width: 920px;
      margin: 18px 0 18px;
      padding: 14px 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .toc h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: .10em;
      color: var(--muted);
    }}
    .toc ul {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 10px;
    }}
    .toc a {{
      color: var(--text);
      text-decoration: none;
      border-bottom: 1px dotted color-mix(in srgb, var(--muted) 70%, transparent);
    }}
    .toc a:hover {{
      color: var(--accent);
      border-bottom-color: var(--accent);
    }}

    .note {{
      text-align: left;
      color: var(--muted);
      font-size: 13px;
      margin: 10px 0 22px;
    }}

    .section {{ margin: 26px 0 10px; }}
    .section > h2 {{
      color: var(--text);
      font-size: 26px;
      letter-spacing: -0.02em;
      margin-bottom: 10px;
    }}

    /* Tables */
    .wrap {{
      overflow-x: auto;
      margin: 0 auto;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    table {{
      border-collapse: separate;
      border-spacing: 0;
      width: 100%;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      white-space: nowrap;
    }}
    th {{
      cursor: pointer;
      position: sticky;
      top: 0;
      background: color-mix(in srgb, var(--panel) 75%, var(--bg));
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      z-index: 2;
    }}
    tbody tr:nth-child(even) td {{
      background: color-mix(in srgb, var(--panel) 70%, var(--bg));
    }}
    tbody tr:hover td {{
      background: color-mix(in srgb, var(--accent) 10%, var(--panel));
    }}

    /* Sticky first column */
    td:first-child, th:first-child {{
      position: sticky;
      left: 0;
      background: inherit;
      z-index: 3;
    }}
    th:first-child {{ z-index: 4; }}

    /* MathJax spacing */
    .metrics mjx-container[display="true"] {{ margin: 0.6em 0; }}
    .metrics td {{ vertical-align: top; white-space: normal; }}

    /* Back to top */
    .toTop {{
      position: fixed;
      right: 18px;
      bottom: 18px;
      opacity: 0;
      pointer-events: none;
      transform: translateY(6px);
      transition: opacity .2s ease, transform .2s ease;
      z-index: 60;
    }}
    .toTop.show {{
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0);
    }}

    .ref a {{
      color: inherit;
      text-decoration: none;
      border-bottom: 1px dotted color-mix(in srgb, var(--muted) 70%, transparent);
    }}
    .ref a:hover {{
      color: var(--accent);
      border-bottom-color: var(--accent);
    }}
  </style>
</head>

<body>
  <div class="topnav">
    <div class="topnav-inner">
      <div class="navlinks">
        <a href="#datasets">Datasets</a>
        <a href="#metrics">Metrics</a>
        <a href="#tasks">Tasks</a>
        <a href="#methods">Methods</a>
        <a href="#tables">Tables</a>
        <a href="#refs">Refs</a>
      </div>
      <button class="btn" id="themeToggle" title="Toggle theme">🌓 Theme</button>
    </div>
  </div>
  <button class="btn toTop" id="toTop" title="Back to top">↑ Top</button>

  <div class="container">
    <div class="hero">
      <h1>OAXMLC-Bench 𓊳</h1>
      <h2 class="subtitle">Benchmarking Extreme Multi-Label Classification for Semantic Annotation with Multi-Taxonomy Datasets</h2>
      <div class="authors">
        <em>Pietro Caforio*, Christophe Broillet*, Philippe Cudré-Mauroux, Julien Audiffren (University of Fribourg)</em><br>
        <small>*Equal contribution</small>
      </div>
    </div>

    <div class="intro" style="margin-top: 18px;">
      <p>
        Extreme Multi-Label Classification (XMLC) is the task of predicting relevant labels from massive tag sets.
        Many real-world label collections are organized into taxonomies (hierarchical relationships),
        and recent methods show that leveraging taxonomic structure can boost performance.
        However, comprehensive evaluation remains challenging: it is hard to fairly compare methods and disentangle
        the effect of taxonomy from other dataset characteristics.
      </p>

      <p>
        We introduce <b>OAXMLC</b>, a benchmark designed to evaluate how XMLC algorithms leverage taxonomic information
        across multiple tasks:
      </p>

      <ul>
        <li>Classification</li>
        <li>Sub-category analysis</li>
        <li>Completion</li>
        <li>Few-shot learning</li>
      </ul>

      <p>
        We provide two large XMLC datasets extracted from OpenAlex, each featuring two distinct taxonomies over the
        <em>same</em> documents. By benchmarking taxonomy-aware and taxonomy-agnostic methods, we analyze how tasks,
        datasets, and taxonomic properties impact performance.
      </p>
    </div>

    <div class="toc">
      <h2>Contents</h2>
      <ul>
        <li><a href="#datasets">Datasets</a></li>
        <li><a href="#metrics">Metrics</a></li>
        <li><a href="#tasks">Tasks</a></li>
        <li><a href="#methods">Benchmarked methods</a></li>
        <li><a href="#tables">Tables</a></li>
        <li><a href="#refs">References</a></li>
      </ul>
    </div>

    <div class="section" id="datasets"><h2>Datasets</h2></div>
    <div class="datasets">
      <p>
        The benchmark is built on two large-scale XMLC datasets extracted from OpenAlex, each equipped with two
        independent taxonomies over the <em>same</em> documents:
        <b>Topics</b> (ASJC/CWTS-derived; 3 levels after extracting the domain-specific sub-taxonomy) and
        <b>Concepts</b> (MAG-derived; 5 hierarchical levels).
      </p>
      <ul>
        <li><b>OAXMLC-CS</b>: computer science publications (broad scope).</li>
        <li><b>OAXMLC-Med</b>: surgery-related medical publications (narrower scope).</li>
      </ul>

      <div class="wrap">
        <table>
          <thead>
            <tr>
              <th>Dataset / Taxonomy</th>
              <th>Documents (N)</th>
              <th>Labels (N)</th>
              <th>Labels / doc (avg)</th>
              <th>Labels / doc (median)</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><b>OAXMLC-CS</b></td><td>3,725,870</td><td>–</td><td>–</td><td>–</td></tr>
            <tr><td>&nbsp;&nbsp;↳ Topics</td><td>–</td><td>775</td><td>3.6</td><td>3</td></tr>
            <tr><td>&nbsp;&nbsp;↳ Concepts</td><td>–</td><td>8,926</td><td>9.8</td><td>9</td></tr>
            <tr><td><b>OAXMLC-Med</b></td><td>869,402</td><td>–</td><td>–</td><td>–</td></tr>
            <tr><td>&nbsp;&nbsp;↳ Topics</td><td>–</td><td>198</td><td>2.4</td><td>2</td></tr>
            <tr><td>&nbsp;&nbsp;↳ Concepts</td><td>–</td><td>2,453</td><td>4.1</td><td>4</td></tr>
            <tr><td><b>MAG-CS</b></td><td>143,928</td><td>2,641</td><td>4.4</td><td>4</td></tr>
            <tr><td><b>EURLex</b></td><td>51,000</td><td>4,492</td><td>10.4</td><td>10</td></tr>
            <tr><td><b>PubMed</b></td><td>139,932</td><td>5,911</td><td>18.5</td><td>18</td></tr>
          </tbody>
        </table>
      </div>
      <p style="text-align:left; color: var(--muted); font-size: 13px; margin-top: 8px;">
        XMLC dataset statistics (Table&nbsp;1 in the paper).
      </p>
    </div>

    {metrics_formulas}
    {tasks_section}

    <div class="section" id="methods"><h2>Benchmarked methods</h2></div>
    <div class="methods">
      <div class="wrap">
        <table>
          <thead>
            <tr><th>Method</th><th>Venue / Publication</th><th>Year</th><th>Algorithm Type</th></tr>
          </thead>
          <tbody>
            <tr><td><a href="#ref1">MATCH</a></td><td>WWW</td><td>2021</td><td>Deep learning, taxonomy-aware (Transformer)</td></tr>
            <tr><td><a href="#ref2">XML-CNN</a></td><td>SIGIR</td><td>2017</td><td>Deep learning (CNN-based)</td></tr>
            <tr><td><a href="#ref3">AttentionXML</a></td><td>NeurIPS</td><td>2019</td><td>Deep learning, label-tree attention</td></tr>
            <tr><td><a href="#ref4">FastXML</a></td><td>KDD</td><td>2016</td><td>Tree-based, non-deep learning</td></tr>
            <tr><td><a href="#ref5">HECTOR</a></td><td>WWW</td><td>2024</td><td>Deep learning, taxonomy-aware (Seq2Seq)</td></tr>
            <tr><td><a href="#ref6">TAMLEC</a></td><td>CIKM / arXiv</td><td>2024–2025</td><td>Deep learning, taxonomy-aware (parallel / path-based)</td></tr>
            <tr><td><a href="#ref7">LightXML</a></td><td>AAAI</td><td>2021</td><td>Deep learning (Transformer, negative sampling)</td></tr>
            <tr><td><a href="#ref8">CascadeXML</a></td><td>NeurIPS</td><td>2022</td><td>Deep learning (multi-resolution Transformer)</td></tr>
            <tr><td><a href="#ref9">Parabel</a></td><td>WWW</td><td>2018</td><td>Tree-based, embedding-based</td></tr>
            <tr><td><a href="#ref10">NGAME</a></td><td>WSDM</td><td>2023</td><td>Deep learning, Siamese / metric learning</td></tr>
            <tr><td><a href="#ref11">DEXA</a></td><td>KDD</td><td>2023</td><td>Deep learning, Siamese with auxiliary parameters</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="note">Tip: click any column header to sort (numeric-aware). The first column (“Method”) stays visible while scrolling.</div>

    <div class="section" id="tables"><h2>Tables</h2></div>
    {content}

    <div class="section" id="refs"><h2>References</h2></div>
    <div class="ref">
      <p id="ref1"><b>[1]</b> Zhang, Y., Shen, Z., Dong, Y., Wang, K., &amp; Han, J. (2021, April).
        <i>MATCH: Metadata-aware text classification in a large hierarchy.</i>
        In Proceedings of the Web Conference 2021 (pp. 3246–3257).</p>

      <p id="ref2"><b>[2]</b> Liu, J., Chang, W. C., Wu, Y., &amp; Yang, Y. (2017, August).
        <i>Deep learning for extreme multi-label text classification.</i>
        In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 115–124).</p>

      <p id="ref3"><b>[3]</b> You, R., Zhang, Z., Wang, Z., Dai, S., Mamitsuka, H., &amp; Zhu, S. (2019).
        <i>AttentionXML: Label tree-based attention-aware deep model for high-performance extreme multi-label text classification.</i>
        Advances in Neural Information Processing Systems, 32.</p>

      <p id="ref4"><b>[4]</b> Jain, H., Prabhu, Y., &amp; Varma, M. (2016, August).
        <i>Extreme multi-label loss functions for recommendation, tagging, ranking &amp; other missing label applications.</i>
        In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 935–944).</p>

      <p id="ref5"><b>[5]</b> Ostapuk, N., Audiffren, J., Dolamic, L., Mermoud, A., &amp; Cudré-Mauroux, P. (2024, May).
        <i>Follow the Path: Hierarchy-Aware Extreme Multi-Label Completion for Semantic Text Tagging.</i>
        In Proceedings of the ACM on Web Conference 2024 (pp. 2094–2105).</p>

      <p id="ref6"><b>[6]</b> Audiffren, J., Broillet, C., Dolamic, L., &amp; Cudré-Mauroux, P. (2024).
        <i>Extreme Multi-label Completion for Semantic Document Labelling with Taxonomy-Aware Parallel Learning.</i>
        arXiv preprint arXiv:2412.13809.</p>

      <p id="ref7"><b>[7]</b> Jiang, T., Wang, D., Sun, L., Yang, H., Zhao, Z., &amp; Zhuang, F. (2021, May).
        <i>LightXML: Transformer with dynamic negative sampling for high-performance extreme multi-label text classification.</i>
        In Proceedings of the AAAI Conference on Artificial Intelligence, 35(9), 7987–7994.</p>

      <p id="ref8"><b>[8]</b> Kharbanda, S., Banerjee, A., Schultheis, E., &amp; Babbar, R. (2022).
        <i>CascadeXML: Rethinking transformers for end-to-end multi-resolution training in extreme multi-label classification.</i>
        Advances in Neural Information Processing Systems, 35, 2074–2087.</p>

      <p id="ref9"><b>[9]</b> Prabhu, Y., Kag, A., Harsola, S., Agrawal, R., &amp; Varma, M. (2018, April).
        <i>Parabel: Partitioned label trees for extreme classification with application to dynamic search advertising.</i>
        In Proceedings of the 2018 World Wide Web Conference (pp. 993–1002).</p>

      <p id="ref10"><b>[10]</b> Dahiya, K., Gupta, N., Saini, D., Soni, A., Wang, Y., Dave, K., Jiao, J., Gururaj, K., Dey, P., Singh, A., Hada, D., Jain, V., Paliwal, B., Mittal, A., Mehta, S., Ramjee, R., Agarwal, S., Kar, P., &amp; Varma, M. (2023, March).
        <i>NGAME: Negative mining-aware mini-batching for extreme classification.</i>
        In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (pp. 258–266).</p>

      <p id="ref11"><b>[11]</b> Dahiya, K., Yadav, S., Sondhi, S., Saini, D., Mehta, S., Jiao, J., Agarwal, S., Kar, P., &amp; Varma, M. (2023).
        <i>Deep Encoders with Auxiliary Parameters for Extreme Classification.</i>
        In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), 358–367.</p>
    </div>
  </div>

  <script>
  // ---------- sorting ----------
  function xcParse(v) {{
    const s = String(v).trim();
    const n = Number(s);
    return (s !== "" && Number.isFinite(n)) ? n : s.toLowerCase();
  }}
  function xcSort(tableId, col) {{
    const table = document.getElementById(tableId);
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const th = table.tHead.rows[0].cells[col];

    const dir = (th.dataset.sort === "asc") ? "desc" : "asc";
    Array.from(table.tHead.rows[0].cells).forEach(h => h.dataset.sort = "none");
    th.dataset.sort = dir;

    rows.sort((a,b) => {{
      const av = xcParse(a.cells[col].textContent);
      const bv = xcParse(b.cells[col].textContent);
      if (av < bv) return dir === "asc" ? -1 : 1;
      if (av > bv) return dir === "asc" ? 1 : -1;
      return 0;
    }});
    rows.forEach(r => tbody.appendChild(r));
  }}
  document.addEventListener("click", (e) => {{
    const th = e.target.closest("th[data-col]");
    if (!th) return;
    const table = th.closest("table");
    xcSort(table.id, Number(th.dataset.col));
  }});

  // ---------- theme toggle ----------
  (function(){{
    const btn = document.getElementById("themeToggle");
    const root = document.documentElement;
    const key = "oaxmlc_theme";

    function apply(mode) {{
      if (!mode) {{
        root.removeAttribute("data-theme");
        return;
      }}
      root.setAttribute("data-theme", mode);
    }}

    const saved = localStorage.getItem(key);
    if (saved) apply(saved);

    btn?.addEventListener("click", () => {{
      const cur = localStorage.getItem(key);
      const next = (cur === "dark") ? "light" : "dark";
      localStorage.setItem(key, next);
      apply(next);
    }});
  }})();

  // ---------- back to top ----------
  (function(){{
    const b = document.getElementById("toTop");
    const onScroll = () => {{
      if (window.scrollY > 600) b.classList.add("show");
      else b.classList.remove("show");
    }};
    window.addEventListener("scroll", onScroll, {{passive:true}});
    onScroll();
    b?.addEventListener("click", () => window.scrollTo({{top:0, behavior:"smooth"}}));
  }})();
  </script>

  <!-- MathJax for LaTeX-like content -->
  <script>
  window.MathJax = {{
    tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']] }},
    options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }}
  }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</body>
</html>
"""


# -----------------------------
# Helpers
# -----------------------------

def title_from_filename(name: str) -> str:
    if name in TABLE_TITLES:
        return TABLE_TITLES[name]
    stem = name[:-4]
    stem = stem.replace("table_", "")
    parts = stem.split("_")
    if len(parts) >= 2:
        task = parts[0]
        dataset = "_".join(parts[1:])
        task = {
            "classification": "Classification",
            "classification_level": "Classification (per level)",
            "classification_levels": "Classification (per level)",
            "completion": "Completion",
            "fewshot": "Few-shot",
            "fewshot_task": "Few-shot",
        }.get(task, task.replace("-", " ").title())
        dataset = dataset.replace("oaxmlc", "OAXMLC").replace("oamed", "OAMED").replace("_", " ").title()
        return f"{task} — {dataset}"
    return name


def parse_table_metadata(name: str):
    """
    Extract (dataset, experiment) from a CSV filename.

    Expected patterns:
      table_<experiment>_<dataset>.csv

    where dataset is one of:
      oaxmlc_concepts, oaxmlc_topics, oamedconcepts, oamedtopics
    """
    stem = name[:-4] if name.endswith(".csv") else name
    stem = stem.replace("table_", "")

    m = re.search(r"(oaxmlc[_]?concepts|oaxmlc[_]?topics|oamedconcepts|oamedtopics)$", stem)
    if m:
        dataset_raw = m.group(1)
        experiment_key = stem[:m.start()].rstrip("_")
    else:
        parts = stem.split("_")
        dataset_raw = parts[-1] if len(parts) > 1 else ""
        experiment_key = "_".join(parts[:-1]) if len(parts) > 1 else stem

    dataset_key = DATASET_ALIASES.get(dataset_raw, dataset_raw)
    dataset_label = DATASET_PRETTY.get(
        dataset_key,
        dataset_raw.replace("oaxmlc", "OAXMLC").replace("oamed", "OAMED").replace("_", " ").title()
        if dataset_raw else "Unknown Dataset",
    )
    experiment_label = EXPERIMENT_PRETTY.get(
        experiment_key,
        experiment_key.replace("-", " ").replace("_", " ").title() if experiment_key else "Unknown Experiment",
    )
    return dataset_key, dataset_label, experiment_key, experiment_label


def dataset_sort_key(dataset_key: str, dataset_label: str):
    try:
        idx = DATASET_ORDER.index(dataset_key)
    except ValueError:
        idx = len(DATASET_ORDER)
    return (idx, dataset_label)


def experiment_sort_key(experiment_key: str, experiment_label: str):
    try:
        idx = EXPERIMENT_ORDER.index(experiment_key)
    except ValueError:
        idx = len(EXPERIMENT_ORDER)
    return (idx, experiment_label)


def pretty_method(x: str) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    key = s.lower()
    key = re.sub(r"[^a-z0-9\-]+", "", key)
    return ALG_PRETTY.get(key, s)


def pretty_metric(name: str) -> str:
    return METRIC_PRETTY.get(name, name)


def _fmt_cell(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (float, int)) and not isinstance(x, bool):
        return f"{x:.4f}" if isinstance(x, float) else str(x)
    return str(x)


def df_to_table_html(df: pd.DataFrame, table_id: str, caption: str | None = None) -> str:
    """
    Convert a DataFrame to a sortable HTML table.
    - Escapes all cell values.
    - Pretty method names in first column.
    - Optional caption displayed above table.
    """
    df = df.copy()
    df = df.rename(columns={df.columns[0]: "Method"})
    df["Method"] = df["Method"].map(pretty_method)

    cols = list(df.columns)

    thead = "<tr>" + "".join(
        f'<th data-col="{i}" data-sort="none">{html.escape(pretty_metric(str(c)))}</th>'
        for i, c in enumerate(cols)
    ) + "</tr>"

    rows = []
    for _, r in df.iterrows():
        tds = "".join(f"<td>{html.escape(_fmt_cell(r[c]))}</td>" for c in cols)
        rows.append(f"<tr>{tds}</tr>")
    tbody = "\n".join(rows)

    cap = f'<div class="note" style="margin: 10px 0 10px;">{html.escape(caption)}</div>' if caption else ""
    return (
        f"{cap}"
        f'<div class="wrap">'
        f'<table id="{table_id}">'
        f"<thead>{thead}</thead>"
        f"<tbody>{tbody}</tbody>"
        f"</table>"
        f"</div>"
    )


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True, help="Folder containing CSVs")
    ap.add_argument("--out", type=Path, required=True, help="Output HTML file (e.g., docs/index.html)")
    args = ap.parse_args()

    csvs = sorted(args.in_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV files found in {args.in_dir}")

    entries = []
    for p in csvs:
        dataset_key, dataset_label, experiment_key, experiment_label = parse_table_metadata(p.name)
        entries.append(
            {
                "path": p,
                "dataset_key": dataset_key,
                "dataset_label": dataset_label,
                "experiment_key": experiment_key,
                "experiment_label": experiment_label,
            }
        )

    entries.sort(
        key=lambda e: (
            dataset_sort_key(e["dataset_key"], e["dataset_label"]),
            experiment_sort_key(e["experiment_key"], e["experiment_label"]),
            e["path"].name,
        )
    )

    parts = []
    current_dataset = None
    current_experiment = None
    table_idx = 0

    for entry in entries:
        if entry["dataset_label"] != current_dataset:
            parts.append(f"<h3>{html.escape(entry['dataset_label'])}</h3>")
            current_dataset = entry["dataset_label"]
            current_experiment = None

        if entry["experiment_label"] != current_experiment:
            parts.append(f"<h4>{html.escape(entry['experiment_label'])}</h4>")
            current_experiment = entry["experiment_label"]

        df = pd.read_csv(entry["path"])
        table_id = f"t{table_idx}"
        caption = title_from_filename(entry["path"].name)
        parts.append(df_to_table_html(df, table_id, caption=None))  # keep clean; enable if you want captions
        table_idx += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    page = HTML_PAGE.format(
        content="\n".join(parts),
        metrics_formulas=METRICS_FORMULAS_HTML,
        tasks_section=TASKS_HTML,
    )
    args.out.write_text(page, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
