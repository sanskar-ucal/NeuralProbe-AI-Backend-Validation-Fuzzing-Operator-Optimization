"""Diagnostic reports: Markdown (GitHub step summary / PR comment), standalone HTML, and JSON."""
from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path

STATUSES = ("PASS", "WARN", "FAIL", "UNSUPPORTED")


def _fmt(v, spec=".3g"):
    if v is None:
        return "–"
    return format(v, spec) if isinstance(v, float) else str(v)


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return "\n".join(out)


def correctness_sections(fuzz: dict) -> list[str]:
    rows = fuzz["rows"]
    backends = [b["name"] for b in fuzz["backends"]]
    ops = list(dict.fromkeys(r["op"] for r in rows))
    out = []

    cells = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = "FAIL*" if r["status"] == "FAIL" and r.get("waived") else r["status"]
        cells[(r["op"], r["backend"])][key] += 1
    mrows = []
    for op in ops:
        line = [f"`{op}`"]
        for b in backends:
            c = cells.get((op, b))
            if not c:
                line.append("–")
                continue
            parts = [f"{c['PASS']}✓"]
            if c["WARN"]:
                parts.append(f"{c['WARN']}⚠")
            if c["FAIL"]:
                parts.append(f"**{c['FAIL']}✗**")
            if c["FAIL*"]:
                parts.append(f"{c['FAIL*']}✗(waived)")
            if c["UNSUPPORTED"]:
                parts.append(f"{c['UNSUPPORTED']} unsup")
            line.append(" ".join(parts))
        mrows.append(line)
    out.append("## Correctness matrix\n\n✓ pass · ⚠ precision warning · ✗ fail · unsup = backend rejected the op\n\n"
               + _table(["op"] + [f"`{b}`" for b in backends], mrows))

    fails = [r for r in rows if r["status"] == "FAIL" and not r.get("waived")]
    if fails:
        frows = [[f"`{r['op']}`", f"`{r['backend']}`", f"{r['dtype']} {r['quant']} {r['dist']}", r["category"],
                  r["label"], (r.get("detail") or "")[:160], r.get("triage") or "–",
                  f"`{r['repro']}`" if r.get("repro") else "–",
                  f"`{r['minimized_label']}`" if r.get("minimized_label") else "–"] for r in fails[:50]]
        out.append(f"## ❌ Unwaived failures ({len(fails)})\n\n"
                   + _table(["op", "backend", "dtype/quant/dist", "category", "shape", "detail", "auto-triage",
                             "repro", "minimized"], frows))
    else:
        out.append("## ✅ No unwaived correctness failures")

    waived = [r for r in rows if r["status"] == "FAIL" and r.get("waived")]
    if waived:
        groups = defaultdict(list)
        for r in waived:
            groups[(r["op"], r["backend"], r["category"], r.get("waiver_reason", ""))].append(r)
        wrows = [[f"`{op}`", f"`{b}`", cat, len(v), reason,
                  next((x["triage"] for x in v if x.get("triage")), "–"),
                  f"`{next((x['repro'] for x in v if x.get('repro')), '–')}`"]
                 for (op, b, cat, reason), v in groups.items()]
        out.append("## Known issues (waived)\n\n" + _table(
            ["op", "backend", "category", "count", "reason", "auto-triage", "example repro"], wrows))

    warns = [r for r in rows if r["status"] == "WARN"]
    if warns:
        wr = [[f"`{r['op']}`", f"`{r['backend']}`", f"{r['dtype']} {r['dist']}", r["label"], r["detail"]]
              for r in sorted(warns, key=lambda r: -(r.get("max_ulp") or 0))[:15]]
        out.append(f"## ⚠ Precision warnings ({len(warns)})\n\nWithin tolerance but much less accurate than PyTorch "
                   "at the same dtype.\n\n" + _table(["op", "backend", "dtype/dist", "shape", "detail"], wr))

    prec: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["status"] in ("PASS", "WARN") and r.get("max_ulp") is not None and r["quant"] == "none":
            p = prec[(r["op"], r["dtype"])]
            p[r["backend"]] = max(p.get(r["backend"], 0.0), r["max_ulp"])
            p["torch"] = max(p.get("torch", 0.0), r.get("torch_max_ulp") or 0.0)
    if prec:
        cols = ["torch"] + backends
        prow = [[f"`{op}`", dt] + [_fmt(v.get(b), ".1f") for b in cols] for (op, dt), v in sorted(prec.items())]
        out.append("## Precision profile\n\nMax condition-normalized ULP error vs the fp64 oracle over passing "
                   "float cases (`torch` = PyTorch at the same dtype).\n\n"
                   + _table(["op", "dtype"] + [f"`{b}`" for b in cols], prow))

    unsup = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["status"] == "UNSUPPORTED":
            unsup[r["backend"]][f"{r['op']} {r['dtype']} {r['quant']}"] += 1
    if unsup:
        urows = [[f"`{b}`", ", ".join(f"{k} ({n})" for k, n in sorted(v.items()))] for b, v in unsup.items()]
        out.append("## Backend coverage gaps (UNSUPPORTED)\n\n" + _table(["backend", "op dtype quant (count)"], urows))
    return out


def perf_sections(bench: dict, regressions: list[dict] | None) -> list[str]:
    out = []
    results = bench["results"]
    backends = list(dict.fromkeys(b for r in results for b in r["backends"]))
    prow = []
    for r in results:
        line = [r["key"]]
        for b in backends:
            v = r["backends"].get(b)
            if not v:
                line.append("–")
            elif "median_us" not in v:
                line.append(v.get("status", "–"))
            else:
                mark = " 🏆" if r.get("analysis", {}).get("fastest") == b else ""
                bad = " ❗" if not v.get("correct", True) else ""
                line.append(f"{v['median_us']:.1f} µs ({v['gflops']:.1f} GF/s){mark}{bad}")
        prow.append(line)
    out.append(f"## Performance (median latency, {bench['threads']} thread(s))\n\n"
               + _table(["benchmark"] + [f"`{b}`" for b in backends], prow))

    findings = [(r["key"], f) for r in results for f in r.get("analysis", {}).get("findings", [])]
    if findings:
        out.append("## Bottleneck analysis\n\n" + "\n".join(f"- **{k}**: {f}" for k, f in findings))

    if regressions is not None:
        bad = [r for r in regressions if r["status"] == "REGRESSION"]
        good = [r for r in regressions if r["status"] == "IMPROVEMENT"]
        title = f"## {'❌' if bad else '✅'} Perf regression check: {len(bad)} regression(s), {len(good)} improvement(s)"
        shown = [r for r in regressions if r["status"] != "OK"]
        if shown:
            rrows = [[r["key"], f"`{r['backend']}`", _fmt(r["base_us"], ".1f"), _fmt(r["head_us"], ".1f"),
                      f"{r['ratio']:.2f}x" if r["ratio"] else "–", r["status"]] for r in shown]
            out.append(title + "\n\n" + _table(["benchmark", "backend", "base µs", "head µs", "ratio", "status"], rrows))
        else:
            out.append(title + "\n\nAll benchmarks within threshold.")
    return out


def render_markdown(summary: dict) -> str:
    parts = [f"# NeuralProbe report — {summary['config']['name']}"]
    verdict = "❌ FAIL" if summary["failed"] else "✅ PASS"
    env = summary["environment"]
    head = [f"**Verdict: {verdict}**", "",
            f"- seed `{summary['config']['seed']}`, {env['machine']} · {env['platform']}",
            f"- torch {env['torch']} · onnxruntime {env['onnxruntime']} · native ISA `{summary.get('native_isa', '–')}`"]
    if summary.get("fuzz"):
        rows = summary["fuzz"]["rows"]
        n_cases = len({r["case_id"] for r in rows})
        head.append(f"- {n_cases} generated cases → {len(rows)} backend executions in {summary['fuzz']['elapsed_s']}s")
        for name, why in summary["fuzz"].get("skipped_backends", {}).items():
            head.append(f"- backend `{name}` skipped: {why}")
    for reason in summary.get("fail_reasons", []):
        head.append(f"- **gate:** {reason}")
    parts.append("\n".join(head))
    if summary.get("fuzz"):
        parts += correctness_sections(summary["fuzz"])
    if summary.get("bench"):
        parts += perf_sections(summary["bench"], summary.get("regressions"))
    return "\n\n".join(parts) + "\n"


def _md_inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def markdown_to_html(md: str, title: str) -> str:
    lines, body, i = md.splitlines(), [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl.append(lines[i])
                i += 1
            cells = [re.split(r"(?<!\\)\|", t.strip()[1:-1]) for t in tbl if not re.match(r"^\|(-+\|)+$", t)]
            h = "".join(f"<th>{_md_inline(c.strip())}</th>" for c in cells[0])
            rows = "".join("<tr>" + "".join(f"<td>{_md_inline(c.strip().replace(chr(92) + '|', '|'))}</td>"
                                            for c in r) + "</tr>" for r in cells[1:])
            body.append(f"<table><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table>")
            continue
        m = re.match(r"^(#+) (.*)", ln)
        if m:
            body.append(f"<h{len(m.group(1))}>{_md_inline(m.group(2))}</h{len(m.group(1))}>")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_md_inline(lines[i][2:])}</li>")
                i += 1
            body.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip():
            body.append(f"<p>{_md_inline(ln)}</p>")
        i += 1
    css = ("body{font-family:-apple-system,Segoe UI,Helvetica,sans-serif;margin:2rem;color:#1b1f24;max-width:1400px}"
           "table{border-collapse:collapse;margin:1rem 0;font-size:13px}th,td{border:1px solid #d0d7de;padding:4px 8px;"
           "text-align:left}th{background:#f6f8fa}tr:nth-child(even){background:#fafbfc}code{background:#eff1f3;"
           "padding:1px 4px;border-radius:4px;font-size:12px}h2{border-bottom:1px solid #d0d7de;padding-bottom:4px}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{css}</style></head><body>{''.join(body)}</body></html>")


def write_reports(summary: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(summary)
    paths = {"markdown": out_dir / "report.md", "html": out_dir / "report.html", "json": out_dir / "summary.json"}
    paths["markdown"].write_text(md)
    paths["html"].write_text(markdown_to_html(md, f"NeuralProbe — {summary['config']['name']}"))
    paths["json"].write_text(json.dumps(summary, indent=2, default=str))
    return paths
