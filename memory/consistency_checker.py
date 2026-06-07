import re


def check_numbers(sections, result_summary):
    issues = []
    actual_metrics = result_summary.get("metrics", [])
    actual_numbers = set()
    for m in actual_metrics:
        actual_numbers.add(str(round(m["mean"], 2)))
        actual_numbers.add(str(round(m["mean"] * 100, 1)))
        actual_numbers.add(str(round(m["mean"] * 100, 2)))

    number_pattern = re.compile(r'\b\d+\.\d+\b')
    for section_name in ["abstract", "results", "discussion", "conclusion"]:
        content = sections.get(section_name, "")
        numbers_found = number_pattern.findall(content)
        for num in numbers_found:
            val = float(num)
            if 50 < val < 100:
                close = any(
                    abs(val - float(n)) < 2.0
                    for n in actual_numbers if n
                )
                if not close and len(actual_numbers) > 0:
                    issues.append({
                        "type": "number_mismatch",
                        "section": section_name,
                        "value": num,
                        "message": num + "% in " + section_name + " does not match actual metrics"
                    })
    return issues


def check_contributions(sections):
    issues = []
    intro = sections.get("introduction", "")
    methodology = sections.get("methodology", "")

    contrib_count = len(re.findall(r'\(\d+\)|\d+\)', intro))
    if contrib_count == 0:
        issues.append({
            "type": "missing_contributions",
            "section": "introduction",
            "message": "No numbered contributions list found in introduction"
        })

    if len(methodology) < 500:
        issues.append({
            "type": "thin_methodology",
            "section": "methodology",
            "message": "Methodology is " + str(len(methodology)) + " chars — needs 2000+"
        })
    return issues


def check_citations(sections, paper_list):
    issues = []
    paper_titles = [p["title"].lower() for p in paper_list.get("papers", [])]
    full_text = " ".join([
        sections.get("introduction", ""),
        sections.get("related_work", ""),
        sections.get("methodology", ""),
        sections.get("discussion", "")
    ]).lower()

    citation_pattern = re.compile(r'\[(\d+)\]')
    cited_numbers = set(citation_pattern.findall(full_text))
    max_ref = len(paper_titles)

    for num in cited_numbers:
        if int(num) > max_ref:
            issues.append({
                "type": "invalid_citation",
                "section": "unknown",
                "message": "Citation [" + num + "] exceeds available papers (" + str(max_ref) + ")"
            })
    return issues


def check_section_lengths(sections):
    issues = []
    minimums = {
        "abstract": 600,
        "introduction": 1500,
        "related_work": 1500,
        "methodology": 2000,
        "experiments": 1500,
        "results": 1500,
        "discussion": 1500,
        "conclusion": 800,
        "limitations": 600
    }
    for section, min_len in minimums.items():
        content = sections.get(section, "")
        if len(content) < min_len:
            issues.append({
                "type": "short_section",
                "section": section,
                "message": section + " is " + str(len(content)) + " chars, needs " + str(min_len) + "+"
            })
    return issues


def run_all_checks(sections, result_summary, paper_list):
    print("[ConsistencyChecker] Running checks...")
    all_issues = []
    all_issues.extend(check_numbers(sections, result_summary))
    all_issues.extend(check_contributions(sections))
    all_issues.extend(check_citations(sections, paper_list))
    all_issues.extend(check_section_lengths(sections))

    if all_issues:
        print("[ConsistencyChecker] Found " + str(len(all_issues)) + " issues:")
        for issue in all_issues:
            print("  [" + issue["type"].upper() + "] " + issue["section"] + ": " + issue["message"])
    else:
        print("[ConsistencyChecker] All checks passed.")

    return all_issues