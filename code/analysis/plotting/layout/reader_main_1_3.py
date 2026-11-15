#!/usr/bin/env python3
"""Pure-artist layout pass for main Figures 1–3."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
OUT=args.out
DATA=args.data
sys.path.insert(0,str(HERE))

import figure_core_main as core
import layout_style as layout

# Keep the output contract explicit in this entry point; values match the shared helper.
width_mm = 183
tiff_dpi = 600
layout.WIDTH_MM = width_mm
layout.DPI = tiff_dpi
layout.mpl.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

captured = {}
def collect(fig, _out, name, height_mm):
    # Capture scientific-coordinate baseline immediately after original drawing.
    captured[name] = {"fig": fig, "height_mm": height_mm,
                      "baseline": layout.signature(fig)}

core.savefig = collect
sources = core.read_sources(DATA)
summary = {
    "Figure_1": core.figure1(sources, OUT),
    "Figure_2": core.figure2(sources, OUT),
    "Figure_3": core.figure3(sources, OUT),
}
OUT.mkdir(parents=True, exist_ok=True)

# Apply the final type scale before choosing page geometry.
for entry in captured.values():
    layout.bold_all(entry["fig"])

# Figure 1: two compact, wide rows.
fig = captured["Figure_1"]["fig"]
a, b = fig.axes
fig.set_size_inches(layout.WIDTH_MM * layout.MM, 136 * layout.MM, forward=True)
a.set_position([0.17, 0.60, 0.815, 0.32])
b.set_position([0.17, 0.14, 0.815, 0.32])
for ann in b.texts:
    if hasattr(ann, "xy") and hasattr(ann, "xyann") and abs(float(ann.xy[0]) - 15) < 1e-6:
        ann.set_position((10, -35))
layout.remove_old_panel_labels(fig)
layout.label(fig, "A", 0.025, 0.985)
layout.label(fig, "B", 0.025, 0.53)

# Figure 2: larger plot area with a shallow legend/note strip.
fig = captured["Figure_2"]["fig"]
fig.set_size_inches(layout.WIDTH_MM * layout.MM, 155 * layout.MM, forward=True)
ax00, ax01, ax10, ax11 = fig.axes
ax00.set_position([0.20, 0.68, 0.34, 0.235])
ax01.set_position([0.65, 0.68, 0.33, 0.235])
ax10.set_position([0.20, 0.30, 0.34, 0.23])
ax11.set_position([0.65, 0.30, 0.33, 0.23])
for ax, title in [
    (ax00, "Four central-budget\\nconfigurations"),
    (ax01, "Paired budget sensitivity:\\n30 min"),
    (ax10, "Paired budget sensitivity:\\n120 min"),
    (ax11, "Exact country-composition\\npairs"),
]:
    ax._left_title.set_text(title.replace("\\n", "\n"))
ax00.set_xticks([80, 85, 90])
ax01.set_xticks([70, 75, 80, 85])
ax11.set_xticks([60, 70, 80, 90, 100])
for ax in (ax01, ax10, ax11):
    low, high = ax.get_xlim()
    ax.set_xlim(low, high + max((high - low) * 0.12, 0.5))
for ax in (ax00, ax01, ax10, ax11):
    for ann in ax.texts:
        if hasattr(ann, "xy") and hasattr(ann, "xyann"):
            if ax in (ax01, ax10, ax11):
                value = ann.get_text().strip().replace("−", "-")
                numeric = value[1:] if value.startswith("-") else value
                if numeric.replace(".", "", 1).isdigit():
                    ann.set_text(ann.get_text() + " pp")
                    ann.set_position((9, 0))
                else:
                    ann.set_position((7, 0))
            else:
                ann.set_position((7, 0))
            ann.set_ha("left")
for text in fig.texts:
    if text.get_text().startswith("Panel a is the original assignment realization"):
        text.set_text("A: random–clustered difference = 5.58 pp.")
        text.set_position((0.5, 0.07))
if fig.legends:
    fig.legends[0].set_bbox_to_anchor((0.5, 0.15))
layout.remove_old_panel_labels(fig)
layout.label(fig, "A", 0.135, 0.98)
layout.label(fig, "B", 0.585, 0.98)
layout.label(fig, "C", 0.135, 0.60)
layout.label(fig, "D", 0.585, 0.60)

# Figure 3: reduce total height and use the available width for all panels.
fig = captured["Figure_3"]["fig"]
fig.set_size_inches(layout.WIDTH_MM * layout.MM, 196 * layout.MM, forward=True)
a, b, c = fig.axes
a.set_position([0.31, 0.46, 0.67, 0.48])
b.set_position([0.13, 0.095, 0.34, 0.245])
c.set_position([0.64, 0.095, 0.34, 0.245])
a._left_title.set_text("30-minute coverage ranges by country")
a.set_xlabel("Maximum − minimum coverage (pp)")
b.set_ylabel("Coverage range (pp)")
c._left_title.set_text("Allocation contrast\\n(162 countries)".replace("\\n", "\n"))
f3 = summary["Figure_3"]
median = float(f3["median_width_pp"])
p90 = float(f3["p90_width_pp"])
if round(median, 2) != 4.06 or round(p90, 2) != 23.60:
    raise ValueError("Figure 3 coverage-range summaries changed; refusing to relabel lines.")
for text in list(a.texts):
    if text.get_text().startswith("Median "):
        text.remove()
for x, label, color in (
    (median, f"Median {median:.2f} pp", "#B07D3C"),
    (p90, f"90th percentile {p90:.2f} pp", "#75618B"),
):
    a.text(x + 1.0, 0.035, label, transform=a.get_xaxis_transform(),
           ha="center", va="bottom", rotation=90, color=color,
           fontsize=8.5, clip_on=False)
for ann in c.texts:
    if hasattr(ann, "get_text") and ann.get_text().startswith("Central African Republic"):
        if "-29.1 pp" not in ann.get_text():
            raise ValueError("Unexpected CAF annotation value; refusing to alter source content.")
        ann.set_text("Central African\\nRepublic (CAF)\\n−29.1 pp".replace("\\n", "\n"))
        ann.set_position((-28, 25))
        ann.set_ha("left")
    elif hasattr(ann, "get_text") and ann.get_text().startswith("Eswatini (SWZ)"):
        ann.set_position((4, 73))
c.set_xticks([-30, -15, 0, 15, 30])
layout.remove_old_panel_labels(fig)
layout.label(fig, "A", 0.025, 0.985)
layout.label(fig, "B", 0.015, 0.39)
layout.label(fig, "C", 0.53, 0.39)

results = {"figures": {}, "source_summaries": summary}
for name, entry in captured.items():
    height = {"Figure_1": 136, "Figure_2": 155, "Figure_3": 196}[name]
    audit = layout.export(fig=entry["fig"], out=OUT, name=name,
                          height_mm=height, baseline_signature=entry["baseline"])
    results["figures"][name] = {
        "baseline_signature": entry["baseline"],
        "final_signature": audit["artist_signature"],
        "audit": audit,
        "files": [str(OUT / f"{name}.{ext}") for ext in ("pdf", "svg", "png", "tiff")],
    }
(OUT / "main_1_3_signatures.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("SIGNATURE_RESULTS", json.dumps({
    name: {"baseline": item["baseline_signature"],
           "final": item["final_signature"]}
    for name, item in results["figures"].items()
}, ensure_ascii=False), flush=True)
print("AUDIT_RESULTS", json.dumps({
    name: {"outside": len(item["audit"]["outside"]),
           "overlaps": len(item["audit"]["overlaps"]),
           "min_font": item["audit"]["minimum_fontsize"]}
    for name, item in results["figures"].items()
}, ensure_ascii=False), flush=True)


