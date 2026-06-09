import base64
import json
import re
from pathlib import Path
from datetime import datetime
import config


def _img_to_base64(img_path):
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _clean_section(text):
    if not text:
        return ""
    text = re.sub(r'SECTION\s+\d+\s*[-–]\s*[\w\s]+\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'---SECTION---', '', text)
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    def fmt_decimal(m):
        val = float(m.group(0))
        if 0.5 < val < 1.0:
            return "{:.2f}%".format(val*100)
        return m.group(0)
    text = re.sub(r'\b0\.\d{6,}\b', fmt_decimal, text)
    def fmt_std(m):
        val = float(m.group(1))
        return "±{:.2f}%".format(val*100) if val < 0.1 else m.group(0)
    text = re.sub(r'[±]\s*(0\.\d{4,})', fmt_std, text)
    return text.strip()


def _md_to_html(text):
    if not text:
        return ""
    text = _clean_section(text)
    lines = text.split("\n")
    html = []
    in_table = False
    header_done = False
    for line in lines:
        s = line.strip()
        if s.startswith("|"):
            if not in_table:
                html.append('<table class="results-table">')
                in_table = True
                header_done = False
            if re.match(r'^\|[-\s|]+\|$', s):
                header_done = True
                continue
            cells = [c.strip() for c in s.split("|")[1:-1]]
            tag = "th" if not header_done else "td"
            row = "<tr>"
            for c in cells:
                c = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', c)
                row += f"<{tag}>{c}</{tag}>"
            row += "</tr>"
            html.append(row)
            if tag == "th":
                header_done = True
        else:
            if in_table:
                html.append("</table>")
                in_table = False
                header_done = False
            if s.startswith("### "):
                html.append(f"<h3>{s[4:]}</h3>")
            elif s.startswith("## "):
                html.append(f"<h2>{s[3:]}</h2>")
            elif s.startswith("# "):
                html.append(f"<h1>{s[2:]}</h1>")
            elif s == "":
                html.append("<br>")
            else:
                p = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', s)
                p = re.sub(r'\*(.*?)\*', r'<em>\1</em>', p)
                p = re.sub(r'\[(\d+)\]', r'<sup class="cite">[\1]</sup>', p)
                p = re.sub(r'`(.*?)`', r'<code>\1</code>', p)
                html.append(f"<p>{p}</p>")
    if in_table:
        html.append("</table>")
    return "\n".join(html)


def _metrics_table(metrics):
    model_data = {}
    for m in metrics:
        name = m["metric_name"]
        val  = m["mean"]
        std  = m["std"]
        if any(x in name.lower() for x in ["ci","p_value","lower","upper"]):
            continue
        parts = name.split("_")
        model = parts[0]
        metric = "_".join(parts[1:]) if len(parts) > 1 else name
        if model not in model_data:
            model_data[model] = {}
        if 0 < val <= 1.0:
            model_data[model][metric] = "{:.2f}±{:.2f}%".format(val*100, std*100)
        else:
            model_data[model][metric] = "{:.4f}".format(val)

    display = {k:v for k,v in model_data.items() if k.upper() in ["RF","GB","XGB","LGB","RANDOMFOREST","GRADIENTBOOSTING","XGBOOST","LIGHTGBM","XAI"]}
    if not display:
        display = dict(list(model_data.items())[:5])
    if not display:
        return ""

    all_metrics = set()
    for v in display.values():
        all_metrics.update(v.keys())
    priority = ["accuracy","f1_score","precision","recall"]
    ordered = [m for m in priority if m in all_metrics] + [m for m in sorted(all_metrics) if m not in priority]

    name_map = {"RF":"Random Forest","GB":"Gradient Boosting","XGB":"XGBoost","LGB":"LightGBM",
                "RANDOMFOREST":"Random Forest","GRADIENTBOOSTING":"Gradient Boosting",
                "XGBOOST":"XGBoost","LIGHTGBM":"LightGBM","XAI":"XAI Top-5"}

    best = {}
    for metric in ordered:
        vals = []
        for model, data in display.items():
            if metric in data:
                try:
                    v = float(data[metric].split("±")[0].replace("%",""))
                    vals.append((v, model))
                except:
                    pass
        if vals:
            best[metric] = max(vals, key=lambda x: x[0])[1]

    html = '<table class="results-table"><thead><tr><th>Model</th>'
    for m in ordered:
        html += "<th>" + m.replace("_"," ").title() + "</th>"
    html += "</tr></thead><tbody>\n"
    for model, data in display.items():
        dname = name_map.get(model, model)
        html += "<tr><td><strong>" + dname + "</strong></td>"
        for metric in ordered:
            val = data.get(metric, "-")
            html += ("<td><strong>" + val + "</strong></td>") if best.get(metric)==model else ("<td>" + val + "</td>")
        html += "</tr>\n"
    html += "</tbody></table>"
    html += '<p class="table-caption">Table I: Comparison of classification performance (5-fold CV). Bold = best per metric.</p>'
    return html


def export_html(paper_draft, result_summary, title, output_path=None):
    if output_path is None:
        output_path = config.OUTPUTS_FINAL / "paper.html"

    plot_dir   = config.OUTPUTS_PLOTS
    plot_files = sorted(plot_dir.glob("fig*.png"))

    fig_captions = {
        "fig0": "Fig. 1. Proposed system architecture.",
        "fig1": "Fig. 2. All models — grouped metric comparison (5-fold CV).",
        "fig2": "Fig. 3. Accuracy comparison with error bars.",
        "fig3": "Fig. 4. Cross-validation score distribution (box plot).",
        "fig4": "Fig. 5. Radar chart — multi-metric model comparison.",
        "fig5": "Fig. 6. Feature importance and ablation study.",
        "fig6": "Fig. 7. Confusion matrix — best model.",
        "fig7": "Fig. 8. SHAP analysis — feature importance and direction."
    }

    figures_html = ""
    count = 0
    for pf in plot_files:
        b64 = _img_to_base64(pf)
        if not b64:
            continue
        key = pf.stem[:4]
        cap = fig_captions.get(key, "Fig. " + pf.stem)
        figures_html += f'<figure class="paper-figure"><img src="data:image/png;base64,{b64}" alt="{cap}"><figcaption>{cap}</figcaption></figure>\n'
        count += 1

    metrics   = result_summary.get("metrics", [])
    met_table = _metrics_table(metrics)
    findings  = result_summary.get("key_findings", [])
    findings_html = ""
    if findings:
        findings_html = "<h3>Key Findings</h3><ul>"
        for f in findings:
            fc = re.sub(r'\b0\.\d{6,}\b', lambda m: "{:.2f}%".format(float(m.group(0))*100) if 0.5 < float(m.group(0)) < 1.0 else m.group(0), f)
            findings_html += f"<li>{fc}</li>"
        findings_html += "</ul>"

    best_model = result_summary.get("best_model","")
    best_acc   = ""
    for m in metrics:
        if best_model.replace(" ","").lower()[:2] in m["metric_name"].lower() and "accuracy" in m["metric_name"].lower():
            best_acc = "{:.2f}%".format(m["mean"]*100)
            break

    sections_order = ["abstract","introduction","related_work","methodology","experiments","results","discussion","conclusion","limitations"]
    body = ""
    for sec in sections_order:
        content = paper_draft.get(sec,"")
        if not content:
            continue
        sec_title = sec.replace("_"," ").upper()
        body += f'<h2 class="sec-heading">{sec_title}</h2>\n'
        body += _md_to_html(content) + "\n"
        if sec == "results":
            if met_table:
                body += "\n" + met_table + "\n"
            if findings_html:
                body += "\n" + findings_html + "\n"
            if figures_html:
                body += '\n<div class="figures-section">\n' + figures_html + '</div>\n'

    refs = paper_draft.get("references",[])
    if refs:
        body += '<h2 class="sec-heading">REFERENCES</h2>\n<ol class="references">\n'
        for ref in refs:
            clean = ref[ref.find("]")+1:].strip() if ref.startswith("[") else ref
            body += f"<li>{clean}</li>\n"
        body += "</ol>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
/* ── Reset ── */
*{{box-sizing:border-box;margin:0;padding:0;}}

/* ── Page ── */
body{{
    font-family:"Times New Roman",Times,serif;
    font-size:10pt;
    line-height:1.5;
    color:#111;
    background:#e8e8e8;
}}

/* ── IEEE Paper Container ── */
.ieee-paper{{
    max-width:210mm;
    margin:20px auto;
    background:white;
    padding:19mm 19mm 25mm 19mm;
    box-shadow:0 4px 30px rgba(0,0,0,0.2);
}}

/* ── Title Block ── */
.paper-title{{
    font-size:20pt;
    font-weight:bold;
    text-align:center;
    margin-bottom:12px;
    color:#000;
    line-height:1.3;
    font-family:"Times New Roman",Times,serif;
}}
.paper-meta{{
    text-align:center;
    font-size:8.5pt;
    color:#555;
    font-family:Arial,sans-serif;
    margin-bottom:6px;
}}
.paper-badge{{
    text-align:center;
    margin-bottom:14px;
}}
.paper-badge span{{
    background:#1a237e;
    color:white;
    padding:3px 16px;
    border-radius:20px;
    font-size:8pt;
    font-family:Arial,sans-serif;
    letter-spacing:0.8px;
}}
.title-rule{{
    border:none;
    border-top:1.5px solid #1a237e;
    margin:12px 0 16px 0;
}}

/* ── Abstract (full width) ── */
.abstract-block{{
    margin-bottom:16px;
    padding:10px 14px;
    background:#f5f7ff;
    border-left:3px solid #1a237e;
}}
.abstract-label{{
    font-weight:bold;
    font-style:italic;
    font-size:9pt;
    display:inline;
    margin-right:4px;
}}
.abstract-text{{
    font-size:9pt;
    text-align:justify;
    display:inline;
    line-height:1.55;
}}

/* ── Two-Column Layout ── */
.two-col{{
    column-count:2;
    column-gap:5mm;
    column-rule:0.5px solid #ccc;
}}

/* ── Section Headings ── */
h2.sec-heading{{
    font-size:9pt;
    font-weight:bold;
    text-align:center;
    text-transform:uppercase;
    letter-spacing:0.8px;
    margin:14px 0 6px 0;
    color:#000;
    font-family:"Times New Roman",Times,serif;
    border:none;
    break-after:avoid;
    column-span:none;
}}
h3{{
    font-size:9pt;
    font-weight:bold;
    font-style:italic;
    margin:10px 0 4px 0;
    break-after:avoid;
}}

/* ── Body Text ── */
p{{
    font-size:9.5pt;
    text-align:justify;
    margin-bottom:6px;
    line-height:1.5;
    hyphens:auto;
}}
ul,ol{{margin:6px 0 6px 16px;}}
li{{font-size:9.5pt;margin-bottom:3px;line-height:1.5;}}
code{{
    font-family:"Courier New",monospace;
    font-size:8.5pt;
    background:#f4f4f4;
    padding:1px 3px;
    border-radius:2px;
}}
sup.cite{{
    color:#1a237e;
    font-weight:bold;
    font-size:7pt;
}}

/* ── Tables ── */
.results-table{{
    width:100%;
    border-collapse:collapse;
    margin:8px 0 4px 0;
    font-size:8pt;
    font-family:Arial,sans-serif;
    break-inside:avoid;
}}
.results-table th{{
    background:#1a237e;
    color:white;
    padding:5px 6px;
    text-align:center;
    font-weight:bold;
    font-size:8pt;
}}
.results-table td{{
    padding:4px 6px;
    border-bottom:0.5px solid #ddd;
    text-align:center;
    font-size:8pt;
}}
.results-table tr:nth-child(even) td{{background:#f0f3ff;}}
.results-table tr:first-child td{{background:#dde3ff;font-weight:bold;}}
.table-caption{{
    font-size:8pt;
    text-align:center;
    color:#444;
    font-style:italic;
    margin-bottom:8px;
    font-family:Arial,sans-serif;
}}

/* ── Figures ── */
.figures-section{{column-span:all;margin:12px 0;}}
.paper-figure{{
    break-inside:avoid;
    text-align:center;
    margin:10px auto;
    max-width:100%;
}}
.paper-figure img{{
    max-width:100%;
    border:0.5px solid #ccc;
    border-radius:3px;
    box-shadow:0 1px 6px rgba(0,0,0,0.1);
}}
figcaption{{
    font-size:8pt;
    color:#444;
    margin-top:5px;
    font-style:italic;
    text-align:center;
    font-family:Arial,sans-serif;
}}

/* ── References ── */
ol.references{{
    padding-left:16px;
    font-size:8.5pt;
    column-span:all;
}}
ol.references li{{
    margin-bottom:4px;
    line-height:1.4;
    font-size:8.5pt;
}}

/* ── Key findings list ── */
ul li{{font-size:8.5pt;}}

/* ── Print ── */
@media print{{
    body{{background:white;}}
    .ieee-paper{{
        box-shadow:none;
        margin:0;
        padding:15mm 15mm 20mm 15mm;
        max-width:100%;
    }}
    h2.sec-heading{{page-break-after:avoid;}}
    .paper-figure{{page-break-inside:avoid;}}
}}
</style>
</head>
<body>
<div class="ieee-paper">

  <!-- Title Block -->
  <div class="paper-title">{title}</div>
  <div class="paper-meta">Research Agent · Auto-generated · {datetime.now().strftime("%B %d, %Y")}</div>
  {"<div class='paper-badge'><span>" + best_model + " · " + best_acc + " accuracy</span></div>" if best_model and best_acc else ""}
  <hr class="title-rule">

  <!-- Abstract (full width, outside columns) -->
  <div class="abstract-block">
    <span class="abstract-label">Abstract—</span>
    <span class="abstract-text">{_md_to_html(paper_draft.get("abstract","")).replace("<p>","").replace("</p>"," ").strip()}</span>
  </div>

  <!-- Two-column body -->
  <div class="two-col">
    {body}
  </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("[Exporter] IEEE HTML saved: " + str(output_path))
    print("[Exporter] Figures embedded: " + str(count))
    return output_path