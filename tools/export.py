import base64
import json
from pathlib import Path
from datetime import datetime
import config


def _img_to_base64(img_path):
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _md_to_html_basic(text):
    import re
    lines = text.split("\n")
    html = []
    in_table = False
    for line in lines:
        if line.startswith("# "):
            html.append("<h1>" + line[2:] + "</h1>")
        elif line.startswith("## "):
            html.append("<h2>" + line[3:] + "</h2>")
        elif line.startswith("### "):
            html.append("<h3>" + line[4:] + "</h3>")
        elif line.startswith("|"):
            if not in_table:
                html.append('<table class="results-table">')
                in_table = True
            if "---" in line:
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            is_header = html and "<table" in html[-2] if len(html) >= 2 else False
            tag = "th" if is_header else "td"
            row = "<tr>" + "".join(
                "<" + tag + ">" + c.replace("**", "<strong>").replace("**", "</strong>") + "</" + tag + ">"
                for c in cells
            ) + "</tr>"
            html.append(row)
        else:
            if in_table:
                html.append("</table>")
                in_table = False
            if line.startswith("**") and line.endswith("**"):
                html.append("<strong>" + line[2:-2] + "</strong>")
            elif line.startswith("> "):
                html.append("<blockquote>" + line[2:] + "</blockquote>")
            elif line.strip() == "":
                html.append("<br>")
            else:
                line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', line)
                line = re.sub(r'\[(\d+)\]', r'<sup>[\1]</sup>', line)
                html.append("<p>" + line + "</p>")
    if in_table:
        html.append("</table>")
    return "\n".join(html)


def export_html(paper_draft, result_summary, title, output_path=None):
    if output_path is None:
        output_path = config.OUTPUTS_FINAL / "paper.html"

    plot_dir = config.OUTPUTS_PLOTS
    plot_files = sorted(plot_dir.glob("fig*.png"))

    figures_html = ""
    fig_captions = {
        "fig0": "Fig. 1. Proposed system architecture showing the end-to-end pipeline from data acquisition to explainable output.",
        "fig1": "Fig. 2. XAI feature importance analysis using Random Forest. Blue bars indicate top-5 most important features.",
        "fig2": "Fig. 3. Model comparison showing 5-fold cross-validation accuracy with standard deviation error bars.",
        "fig3": "Fig. 4. Cross-validation score distribution across 5 folds for all models (box plot).",
        "fig4": "Fig. 5. Ablation study showing the impact of XAI-based feature selection on classification accuracy.",
        "fig5": "Fig. 6. Confusion matrix for the best-performing model evaluated via 5-fold cross-validation."
    }

    for plot_file in plot_files:
        b64 = _img_to_base64(plot_file)
        if not b64:
            continue
        fig_key = plot_file.stem[:4]
        caption = fig_captions.get(fig_key, "Fig. " + plot_file.stem)
        figures_html += f'''
        <figure class="paper-figure">
            <img src="data:image/png;base64,{b64}" alt="{caption}">
            <figcaption>{caption}</figcaption>
        </figure>
        '''

    metrics = result_summary.get("metrics", [])
    metrics_table = """
    <table class="results-table">
        <thead>
            <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Folds</th></tr>
        </thead>
        <tbody>
    """
    for m in metrics:
        metrics_table += f"<tr><td>{m['metric_name']}</td><td><strong>{m['mean']}</strong></td><td>±{m['std']}</td><td>{m['n_runs']}</td></tr>"
    metrics_table += "</tbody></table>"

    sections_order = ["abstract", "introduction", "related_work", "methodology",
                      "experiments", "results", "discussion", "conclusion", "limitations"]

    body_html = ""
    for i, section in enumerate(sections_order):
        content = paper_draft.get(section, "")
        if not content:
            continue
        section_title = section.replace("_", " ").title()
        body_html += f"<h2>{section_title}</h2>\n"
        body_html += _md_to_html_basic(content) + "\n"
        if section == "results":
            body_html += "<h3>Experimental Results Summary</h3>\n"
            body_html += metrics_table + "\n"
            body_html += "<h3>Figures</h3>\n"
            body_html += figures_html

    refs = paper_draft.get("references", [])
    if refs:
        body_html += "<h2>References</h2>\n<ol class='references'>\n"
        for ref in refs:
            clean = ref[ref.find("]")+1:].strip() if ref.startswith("[") else ref
            body_html += f"<li>{clean}</li>\n"
        body_html += "</ol>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: "Times New Roman", Times, serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #1a1a1a;
            background: #f5f5f5;
        }}
        .paper-container {{
            max-width: 900px;
            margin: 40px auto;
            background: white;
            padding: 60px 80px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.12);
        }}
        h1 {{
            font-size: 18pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 8px;
            color: #1a1a2e;
            line-height: 1.3;
        }}
        .authors {{
            text-align: center;
            font-style: italic;
            margin-bottom: 6px;
            color: #444;
        }}
        .venue-badge {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .venue-badge span {{
            background: #1a1a2e;
            color: white;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 9pt;
            font-family: Arial, sans-serif;
        }}
        .divider {{
            border: none;
            border-top: 2px solid #1a1a2e;
            margin: 20px 0;
        }}
        h2 {{
            font-size: 13pt;
            font-weight: bold;
            margin: 28px 0 10px 0;
            color: #1a1a2e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 4px;
        }}
        h3 {{
            font-size: 12pt;
            font-weight: bold;
            margin: 18px 0 8px 0;
            color: #333;
            font-style: italic;
        }}
        p {{
            margin-bottom: 10px;
            text-align: justify;
        }}
        blockquote {{
            background: #f0f4ff;
            border-left: 4px solid #1a1a2e;
            padding: 10px 16px;
            margin: 12px 0;
            font-style: italic;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 10pt;
            font-family: Arial, sans-serif;
        }}
        .results-table th {{
            background: #1a1a2e;
            color: white;
            padding: 8px 12px;
            text-align: left;
        }}
        .results-table td {{
            padding: 7px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .results-table tr:nth-child(even) {{
            background: #f8f9ff;
        }}
        .results-table tr:first-child td {{
            font-weight: bold;
            background: #e8f0fe;
        }}
        .paper-figure {{
            margin: 24px 0;
            text-align: center;
        }}
        .paper-figure img {{
            max-width: 100%;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        figcaption {{
            font-size: 10pt;
            color: #555;
            margin-top: 8px;
            font-style: italic;
            text-align: center;
        }}
        ol.references {{
            padding-left: 24px;
            font-size: 10pt;
        }}
        ol.references li {{
            margin-bottom: 6px;
            line-height: 1.5;
        }}
        sup {{
            color: #1a1a2e;
            font-weight: bold;
        }}
        .meta-bar {{
            font-family: Arial, sans-serif;
            font-size: 9pt;
            color: #888;
            text-align: center;
            margin-bottom: 20px;
        }}
        @media print {{
            body {{ background: white; }}
            .paper-container {{ box-shadow: none; margin: 0; padding: 20mm; }}
        }}
    </style>
</head>
<body>
<div class="paper-container">
    <h1>{title}</h1>
    <div class="meta-bar">Generated by Research Agent · {datetime.now().strftime("%B %d, %Y")}</div>
    <hr class="divider">
    {body_html}
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("[Exporter] HTML paper saved to: " + str(output_path))
    return output_path