#!/usr/bin/env python
"""Create the five updated main manuscript figures from frozen source tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPGRADE_ROOT = PROJECT_ROOT / "upgrade_20261112"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "figures" / "main"

COLORS = {
    "navy": "#355C7D",
    "teal": "#547A73",
    "ochre": "#B07D3C",
    "plum": "#75618B",
    "slate": "#58636E",
    "gray": "#A8ADB2",
    "light": "#E8EAEC",
    "ink": "#20262C",
    "white": "#FFFFFF",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def require_close(label: str, observed: float, expected: float, atol: float = 1e-6) -> None:
    if not np.isclose(observed, expected, atol=atol, rtol=0):
        raise ValueError(f"{label}: observed {observed:.12g}, expected {expected:.12g}")


def paths_for(project_root: Path, upgrade_root: Path) -> dict[str, Path]:
    work = project_root / "work"
    return {
        "threshold": work / "threshold_curve.csv",
        "scenarios": work / "scenario_budget_final.csv",
        "scenario_json": work / "scenario_budget_final.json",
        "scenario_budget": work / "scenario_budget.csv",
        "u_curve": work / "urban_share_curve.csv",
        "signed_width": work / "identified_set_width.csv",
        "composition_masks": upgrade_root / "server_results" / "derived" / "composition_curve" / "nominal_vs_actual.csv",
        "certified_27": upgrade_root / "derived" / "certification_final" / "certified_27_full_population.csv",
        "strong_widths": upgrade_root / "final_tables" / "stronger_constraint_widths.csv",
        "multiservice": upgrade_root / "final_tables" / "multiservice_spatial_validation.csv",
        "prior_service": upgrade_root / "final_tables" / "prior_service_adjusted_validation.csv",
        "temporal": upgrade_root / "final_tables" / "temporal_validation.csv",
    }


def read_inputs(paths: dict[str, Path]) -> dict[str, object]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required source tables are missing: " + ", ".join(missing))
    return {
        "threshold": pd.read_csv(paths["threshold"]),
        "scenarios": pd.read_csv(paths["scenarios"]),
        "scenario_json": json.loads(paths["scenario_json"].read_text(encoding="utf-8")),
        "budget": pd.read_csv(paths["scenario_budget"]),
        "u_curve": pd.read_csv(paths["u_curve"]),
        "signed": pd.read_csv(paths["signed_width"]),
        "masks": pd.read_csv(paths["composition_masks"]),
        "certified": pd.read_csv(paths["certified_27"]),
        "strong": pd.read_csv(paths["strong_widths"]),
        "multiservice": pd.read_csv(paths["multiservice"]),
        "prior": pd.read_csv(paths["prior_service"]),
        "temporal": pd.read_csv(paths["temporal"]),
    }


def validate_sources(d: dict[str, object]) -> None:
    threshold = d["threshold"]
    expected_defs = {"D0u", "D1", "D2meas_corr", "D1walk"}
    if set(threshold["defn"].astype(str)) != expected_defs:
        raise ValueError("Threshold source must contain D0u, D1, D2meas_corr, and D1walk only.")
    t30 = threshold.loc[threshold["thr_min"].eq(30)].set_index("defn")
    require_close("D1 30-min coverage (%)", float(t30.loc["D1", "all"] * 100), 85.0494564, atol=0.001)
    require_close("D2meas_corr 30-min coverage (%)", float(t30.loc["D2meas_corr", "all"] * 100), 82.18, atol=0.02)

    scenarios = d["scenarios"].set_index("scenario")
    expected_scenarios = {"rand", "urban", "rural", "cluster"}
    if set(scenarios.index) != expected_scenarios or not scenarios.index.is_unique:
        raise ValueError("The final budget table must contain exactly one row for each of four scenarios.")
    for name, expected in {"rand": 83.9774727, "urban": 82.5849771, "rural": 84.5112324, "cluster": 78.3938468}.items():
        require_close(f"{name} 30-min coverage (%)", float(scenarios.loc[name, "c30"] * 100), expected, atol=0.002)

    budget = d["budget"]
    if set(budget["scenario"].astype(str)) != expected_scenarios:
        raise ValueError("Per-country budget source must contain all four final scenarios.")
    for name, rows in budget.groupby("scenario", sort=False):
        if len(rows) != 216 or rows["iso3"].nunique() != 216:
            raise ValueError(f"{name} must contain exactly one budget row per 216 countries.")
        if not rows["n_selected"].astype(int).eq(rows["k_c"].astype(int)).all():
            raise ValueError(f"{name} has a country-level facility-count quota violation.")
        if int(rows["n_selected"].sum()) != 184367:
            raise ValueError(f"{name} must select exactly 184,367 facilities globally.")

    masks = d["masks"]
    if len(masks) != 6 or not masks["reconstructed_source_mask_matches"].astype(bool).all():
        raise ValueError("The composition-curve source must contain six reconstructed source masks.")
    if not masks["actual_q_in_045_068"].astype(bool).isin([True, False]).all():
        raise ValueError("Composition-mask q-interval flags are malformed.")
    if not masks["nominal_mix_u"].is_monotonic_increasing:
        raise ValueError("Composition-mask nominal mixing parameter must be ordered.")

    certified = d["certified"]
    if len(certified) != 27 or not certified["certified"].astype(str).str.lower().isin(["true", "1"]).all():
        raise ValueError("The corrected full-population country table must contain 27 certified rows.")
    if certified[["observed", "width", "gap_max", "gap_min"]].isna().any().any():
        raise ValueError("Certified widths and observed coverage must be complete.")
    require_close("27-country median width (pp)", float(certified["width"].median() * 100), 4.060876761, atol=1e-7)
    require_close("27-country p90 width (pp)", float(certified["width"].quantile(0.9) * 100), 23.603405091, atol=1e-7)
    com = certified.loc[certified["iso"].eq("COM")]
    if len(com) != 1:
        raise ValueError("COM must occur exactly once in the corrected full-population certificate.")
    require_close("Comoros maximum width (pp)", float(com["width"].iloc[0] * 100), 41.230210233, atol=1e-7)
    rho = certified["observed"].corr(certified["width"], method="spearman")
    require_close("Spearman rho, observed coverage vs width", float(rho), -0.76869053, atol=1e-7)

    signed = d["signed"]
    if len(signed) != 162 or signed[["C_rural", "C_urban"]].isna().any().any():
        raise ValueError("The signed endpoint source must contain 162 complete countries.")
    caf = signed.loc[signed["iso3"].eq("CAF")]
    if len(caf) != 1:
        raise ValueError("CAF must occur exactly once in signed endpoint contrasts.")
    caf_contrast = float((caf["C_rural"].iloc[0] - caf["C_urban"].iloc[0]) * 100)
    require_close("CAF C(u=0)-C(u=1) contrast (pp)", caf_contrast, -29.1, atol=0.2)

    strong = d["strong"]
    state = strong.loc[
        strong["threshold_minutes"].eq(30)
        & strong["population_group"].eq("all")
        & strong["folder"].isin(["brazil_state_constraints", "mexico_state_constraints"])
    ]
    expected_widths = {("BRA", "chole"): 26.25, ("BRA", "cesarean"): 30.77,
                       ("BRA", "appendectomy"): 31.05, ("MEX", "chole"): 6.23,
                       ("MEX", "cesarean"): 2.29}
    if len(state) != 5:
        raise ValueError(f"Expected five exact-folder 30-min full-population state rows, found {len(state)}.")
    for key, expected in expected_widths.items():
        row = state.loc[state["country"].eq(key[0]) & state["service"].eq(key[1])]
        if len(row) != 1:
            raise ValueError(f"Expected one state-constraint row for {key}.")
        require_close(f"{key[0]} {key[1]} width lower bound (pp)", float(row["width_lower_bound_pp"].iloc[0]), expected, atol=0.01)

    services = {"chole", "cesarean", "appendectomy"}
    multi = d["multiservice"]
    prior = d["prior"].loc[d["prior"]["scheme"].eq("spatial_blocks")]
    temporal = d["temporal"]
    for name, table in (("population+urban spatial", multi), ("prior-service spatial", prior)):
        if set(table["service"].astype(str)) != services or not table["n_repartitions"].astype(int).eq(50).all():
            raise ValueError(f"{name} must contain all three services and 50 spatial repartitions.")
    if set(temporal["service"].astype(str)) != services:
        raise ValueError("Temporal validation must contain all three services.")
    if not multi[["mean", "lo", "hi"]].notna().all().all():
        raise ValueError("Population+urban validation summaries contain missing values.")


def style_axes(ax, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=COLORS["light"], linewidth=0.55)
    ax.tick_params(length=2.4, color=COLORS["slate"], labelcolor=COLORS["ink"])
    ax.spines["left"].set_color("#68727B")
    ax.spines["bottom"].set_color("#68727B")


def panel_label(ax, label: str) -> None:
    ax.text(-0.065, 1.025, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left", color=COLORS["ink"])


def save_figure(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base = out / name
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), format="tiff", dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def save_source(out: Path, name: str, *frames: tuple[str, pd.DataFrame]) -> None:
    parts = []
    for panel, frame in frames:
        part = frame.copy()
        part.insert(0, "source_panel", panel)
        parts.append(part)
    pd.concat(parts, ignore_index=True, sort=False).to_csv(out / f"{name}_source_data.csv", index=False)


def figure1(d: dict[str, object], out: Path) -> None:
    data = d["threshold"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    ax2 = ax.twinx()
    series = [
        ("D0u", "Any health facility (D0u)", COLORS["navy"], "o", "-"),
        ("D1", "OSM hospital (D1)", COLORS["teal"], "s", "-"),
        ("D2meas_corr", "Budget + local-population filter", COLORS["ochre"], "^", "-"),
        ("D1walk", "Walking (D1walk; separate mode)", COLORS["plum"], "D", "--"),
    ]
    for key, label, color, marker, linestyle in series:
        sub = data.loc[data["defn"].eq(key)].sort_values("thr_min")
        ax.plot(sub["thr_min"], sub["all"] * 100, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.35, markersize=3.3, label=label, zorder=3)
    gap = data.loc[data["defn"].eq("D1")].sort_values("thr_min").copy()
    gap["gap_pp"] = (gap["urban"] - gap["rural"]) * 100
    ax2.plot(gap["thr_min"], gap["gap_pp"], color=COLORS["slate"], marker="P",
             linestyle=":", linewidth=1.25, markersize=3.2,
             label="D1 urban − rural gap (right axis)", zorder=4)
    for minute in (15, 30, 120):
        val = float(gap.loc[gap["thr_min"].eq(minute), "gap_pp"].iloc[0])
        ax2.annotate(f"{val:.1f} pp", (minute, val), xytext=(5, 4),
                     textcoords="offset points", fontsize=6.2, color=COLORS["slate"])
    walk15 = float(data.loc[data["defn"].eq("D1walk") & data["thr_min"].eq(15), "all"].iloc[0] * 100)
    ax.annotate(f"Walking, 15 min: {walk15:.1f}%", xy=(15, walk15), xytext=(39, 23),
                textcoords="data", fontsize=6.2, color=COLORS["ink"],
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#D5D9DC"},
                arrowprops={"arrowstyle": "-", "linewidth": 0.55, "color": COLORS["plum"]})
    ax.axvline(120, color="#7C858D", linewidth=0.7, linestyle="--", zorder=1)
    ax.text(120, 101.0, "120 min", ha="center", va="bottom", fontsize=6.3, color=COLORS["slate"])
    ax.set_title("Coverage by travel-time threshold, facility definition and mode", loc="left", pad=7)
    ax.set_xlabel("Travel-time threshold (min)")
    ax.set_ylabel("Population within threshold (%)")
    ax2.set_ylabel("D1 urban − rural difference (percentage points)", color=COLORS["slate"])
    ax2.tick_params(axis="y", colors=COLORS["slate"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COLORS["slate"])
    ax.set_xticks([15, 30, 45, 60, 90, 120, 180])
    ax.set_xlim(8, 187)
    ax.set_ylim(20, 103)
    ax2.set_ylim(0, 45)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticks([0, 10, 20, 30, 40])
    style_axes(ax)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 0.12),
               ncol=3, fontsize=6.3, columnspacing=1.05, handlelength=1.9)
    fig.text(0.5, 0.065, "D2meas_corr = D2meas AND D1_p85. Walking uses a separate friction surface.",
             ha="center", fontsize=6.3, color=COLORS["ink"])
    fig.text(0.5, 0.025, "Population-weighted raster coverage",
             ha="center", fontsize=6.2, color=COLORS["slate"])
    fig.subplots_adjust(left=0.105, right=0.895, top=0.88, bottom=0.28)
    save_figure(fig, out, "Figure_1")
    source = data.copy()
    source["series_label"] = source["defn"].map({k: v for k, v, *_ in series})
    source["D1_urban_minus_rural_pp"] = np.where(
        source["defn"].eq("D1"), (source["urban"] - source["rural"]) * 100, np.nan)
    source["source_file"] = "work/threshold_curve.csv"
    save_source(out, "Figure_1", ("threshold curves", source))


def figure2(d: dict[str, object], out: Path) -> None:
    scenarios = d["scenarios"].set_index("scenario")
    sjson = d["scenario_json"]
    budget = d["budget"].copy()
    ucurve = d["u_curve"]
    masks = d["masks"].copy()
    order = ["rand", "urban", "rural", "cluster"]
    labels = {"rand": "Random", "urban": "High local-pop.", "rural": "Low local-pop.", "cluster": "Clustered"}
    colors = {"rand": COLORS["navy"], "urban": COLORS["teal"], "rural": COLORS["ochre"], "cluster": COLORS["plum"]}
    summary = pd.DataFrame({
        "scenario": order,
        "scenario_label": [labels[s] for s in order],
        "coverage_30_percent": [float(scenarios.loc[s, "c30"] * 100) for s in order],
        "coverage_120_percent": [float(scenarios.loc[s, "c120"] * 100) for s in order],
        "realized_global_q": [float(sjson["u"][s]) for s in order],
        "facilities_selected": [int(budget.loc[budget["scenario"].eq(s), "n_selected"].sum()) for s in order],
        "countries": [int(budget.loc[budget["scenario"].eq(s), "iso3"].nunique()) for s in order],
    })
    c30 = ucurve.loc[ucurve["thr"].eq(30)].sort_values("u").copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.75), gridspec_kw={"width_ratios": [1.05, 1]})
    panel_label(ax1, "a")
    panel_label(ax2, "b")
    style_axes(ax1)
    style_axes(ax2)
    y = np.arange(len(order))
    for idx, scenario in enumerate(order):
        row = summary.loc[summary["scenario"].eq(scenario)].iloc[0]
        value = float(row["coverage_30_percent"])
        q = float(row["realized_global_q"])
        ax1.barh(idx, value, color=colors[scenario], height=0.55, edgecolor="white", linewidth=0.45, zorder=2)
        ax1.text(value + 1.1, idx, f"{value:.2f}%   q={q:.3f}", va="center", fontsize=6.3, color=COLORS["ink"])
    ax1.set_yticks(y, [labels[s] for s in order])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 105)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_xlabel("30-min population coverage (%)")
    ax1.set_title("Four exact country-budget allocations", loc="left", pad=7)
    constrained_width = float(sjson["constrained"]["width"] * 100)
    global_range = float(summary["coverage_30_percent"].max() - summary["coverage_30_percent"].min())
    ax1.text(0.02, 0.97, f"Global scenario range {global_range:.2f} pp; constrained width lower bound {constrained_width:.2f} pp",
             transform=ax1.transAxes, va="top", ha="left", fontsize=6.2, color=COLORS["ink"],
             bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#D5D9DC"})

    ax2.plot(c30["u"], c30["all"] * 100, color=COLORS["navy"], marker="o", linewidth=1.45,
             markersize=3.2, label="Global C(u)")
    ax2.plot(c30["u"], c30["rural"] * 100, color=COLORS["ochre"], marker="s", linewidth=1.25,
             markersize=3.0, label="Rural C(u)")
    ax2.scatter(masks["nominal_mix_u"], masks["coverage30"] * 100, marker="D", s=24,
                facecolor="white", edgecolor=COLORS["plum"], linewidth=0.9, zorder=4,
                label="Six reconstructed source masks")
    for idx, row in masks.reset_index(drop=True).iterrows():
        offset = (0, 7 if idx % 2 == 0 else -12)
        va = "bottom" if idx % 2 == 0 else "top"
        ax2.annotate(f"q={float(row['actual_global_q']):.4f}",
                     (float(row["nominal_mix_u"]), float(row["coverage30"] * 100)),
                     xytext=offset, textcoords="offset points", ha="center", va=va,
                     fontsize=6.1, color=COLORS["plum"])
    ax2.set_title("C(u) mixing and realized constructions", loc="left", pad=7)
    ax2.set_xlabel("Nominal mixture parameter, u")
    ax2.set_ylabel("30-min population coverage (%)")
    ax2.set_xlim(-0.06, 1.06)
    ax2.set_ylim(57, 89)
    ax2.set_xticks(np.linspace(0, 1, 6))
    ax2.legend(loc="center left", bbox_to_anchor=(0.02,0.35), fontsize=6.1, handlelength=1.7, labelspacing=0.4)
    fig.text(0.5, 0.09, "Each allocation selects the exact country quota k_c in all 216 countries (184,367 facilities total).",
             ha="center", fontsize=6.3, color=COLORS["ink"])
    fig.text(0.5, 0.047, "The 5.58 pp constructed lower bound is conditional on global q  in  [0.45, 0.68]. Panel b uses nominal u on the x axis and labels realized q.",
             ha="center", fontsize=6.1, color=COLORS["slate"])
    fig.subplots_adjust(left=0.14, right=0.985, top=0.88, bottom=0.22, wspace=0.37)
    save_figure(fig, out, "Figure_2")

    budget_source = budget.copy()
    budget_source["source_file"] = "work/scenario_budget.csv"
    summary["source_file"] = "work/scenario_budget_final.csv + work/scenario_budget_final.json"
    curve = c30[["u", "thr", "all", "urban", "rural"]].copy()
    curve["source_file"] = "work/urban_share_curve.csv"
    masks["source_file"] = "upgrade_20261112/server_results/derived/composition_curve/nominal_vs_actual.csv"
    save_source(out, "Figure_2", ("country-level exact budgets", budget_source),
                ("scenario coverage and q", summary), ("C(u) curve", curve), ("six masks, nominal u and realized q", masks))


def figure3(d: dict[str, object], out: Path) -> None:
    certified = d["certified"].copy().sort_values("width", ascending=False).reset_index(drop=True)
    certified["width_pp"] = certified["width"] * 100
    certified["all_facility_coverage_percent"] = certified["observed"] * 100
    certified["rank_width_descending"] = np.arange(1, len(certified) + 1)
    median = float(certified["width_pp"].median())
    p90 = float(certified["width_pp"].quantile(0.90))
    rho = float(certified["observed"].corr(certified["width"], method="spearman"))

    signed = d["signed"].copy()
    signed["contrast_pp"] = (signed["C_rural"] - signed["C_urban"]) * 100
    signed = signed.sort_values("contrast_pp").reset_index(drop=True)
    signed["rank_contrast_ascending"] = np.arange(1, len(signed) + 1)

    fig = plt.figure(figsize=(7.2, 7.35))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.15, 1.45, 1.45], hspace=0.52)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    for ax in (ax1, ax2, ax3):
        style_axes(ax)
    panel_label(ax1, "a")
    panel_label(ax2, "b")
    panel_label(ax3, "c")

    y = np.arange(len(certified))
    ax1.barh(y, certified["width_pp"], color=COLORS["navy"], height=0.68,
             edgecolor="white", linewidth=0.25)
    ax1.set_yticks(y, certified["iso"].astype(str), fontsize=6.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 46)
    ax1.set_xticks([0, 10, 20, 30, 40])
    ax1.axvline(median, color=COLORS["ochre"], linestyle="--", linewidth=0.95,
                label=f"Median {median:.2f} pp")
    ax1.axvline(p90, color=COLORS["plum"], linestyle=":", linewidth=1.0,
                label=f"p90 {p90:.2f} pp")
    ax1.set_xlabel("Certified full-population country width (percentage points)")
    ax1.set_title("27 corrected full-population certificates", loc="left", pad=6)
    ax1.legend(loc="lower right", ncol=2, fontsize=6.2, handlelength=1.6)
    com = certified.loc[certified["iso"].eq("COM")].iloc[0]
    ax1.annotate(f"COM {com['width_pp']:.2f} pp", (com["width_pp"], float(com["rank_width_descending"] - 1)),
                 xytext=(-4, -12), textcoords="offset points", ha="right", va="top",
                 fontsize=6.2, color=COLORS["ink"])

    point_colors = np.where(certified["iso"].eq("COM"), COLORS["ochre"], COLORS["navy"])
    ax2.scatter(certified["all_facility_coverage_percent"], certified["width_pp"], c=point_colors,
                s=20, edgecolor="white", linewidth=0.35, zorder=3)
    ax2.set_xlim(20, 102)
    ax2.set_ylim(-1.5, max(45, float(certified["width_pp"].max()) + 3))
    ax2.set_xticks([30, 50, 70, 90, 100])
    ax2.set_xlabel("All-facility population coverage (%)")
    ax2.set_ylabel("Certified width (pp)")
    ax2.set_title(f"All-facility coverage vs width (Spearman ρ = {rho:.3f})", loc="left", pad=5)

    caf = signed.loc[signed["iso3"].eq("CAF")].iloc[0]
    signed_colors = np.where(signed["iso3"].eq("CAF"), COLORS["plum"], COLORS["gray"])
    ax3.scatter(signed["contrast_pp"], signed["rank_contrast_ascending"], c=signed_colors,
                s=10, linewidth=0, zorder=3)
    ax3.axvline(0, color=COLORS["ink"], linewidth=0.75)
    ax3.set_yticks([])
    xmin = min(-31, float(signed["contrast_pp"].min()) - 1)
    xmax = max(31, float(signed["contrast_pp"].max()) + 1)
    ax3.set_xlim(xmin, xmax)
    ax3.set_xlabel("Signed C(u=0) − C(u=1) endpoint contrast (pp)")
    ax3.set_title("162-country allocation contrast (distinct from set width)", loc="left", pad=5)
    ax3.annotate(f"CAF {caf['contrast_pp']:.1f} pp", (caf["contrast_pp"], caf["rank_contrast_ascending"]),
                 xytext=(5, 4), textcoords="offset points", ha="left", va="bottom",
                 fontsize=6.2, color=COLORS["plum"],
                 arrowprops={"arrowstyle": "-", "linewidth": 0.45, "color": COLORS["plum"]})
    fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.075)
    save_figure(fig, out, "Figure_3")

    certified["source_file"] = "upgrade_20261112/derived/certification_final/certified_27_full_population.csv"
    signed["source_file"] = "work/identified_set_width.csv"
    save_source(out, "Figure_3", ("27 corrected country widths and observed coverage", certified),
                ("162 signed endpoint contrasts", signed))


def observed_value(row: pd.Series) -> float:
    if "observed" in str(row["max_scenario"]):
        return float(row["high_percent"])
    if "observed" in str(row["min_scenario"]):
        return float(row["low_percent"])
    raise ValueError(f"Observed scenario not found among endpoints for {row['country']} {row['service']}.")


def figure4(d: dict[str, object], out: Path) -> None:
    full = d["strong"]
    state = full.loc[
        full["threshold_minutes"].eq(30)
        & full["population_group"].eq("all")
        & full["folder"].isin(["brazil_state_constraints", "mexico_state_constraints"])
    ].copy()
    state["observed_coverage_percent"] = state.apply(observed_value, axis=1)
    service_labels = {"chole": "Cholecystectomy", "cesarean": "Cesarean delivery", "appendectomy": "Appendectomy"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.1), sharex=True,
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    panel_label(ax1, "a")
    panel_label(ax2, "b")
    for ax in (ax1, ax2):
        style_axes(ax)
        ax.set_xlim(40, 96)
        ax.set_xticks([40, 50, 60, 70, 80, 90])
        ax.set_xlabel("30-min population coverage (%)")

    panel_specs = [
        (ax1, "BRA", ["chole", "cesarean", "appendectomy"], "Brazil"),
        (ax2, "MEX", ["chole", "cesarean"], "Mexico"),
    ]
    for ax, country, service_order, country_name in panel_specs:
        rows = state.loc[state["country"].eq(country)].set_index("service")
        yy = np.arange(len(service_order))
        for idx, service in enumerate(service_order):
            row = rows.loc[service]
            lo = float(row["low_percent"])
            hi = float(row["high_percent"])
            ax.hlines(idx, lo, hi, color=COLORS["navy"], linewidth=2.0, zorder=2)
            ax.scatter([lo, hi], [idx, idx], color=COLORS["navy"], s=17, zorder=3,
                       edgecolor="white", linewidth=0.35)
            ax.scatter(float(row["observed_coverage_percent"]), idx, color=COLORS["ochre"],
                       marker="D", s=24, zorder=4, edgecolor="white", linewidth=0.35)
            ax.annotate(f"{float(row['width_lower_bound_pp']):.2f} pp lower bound",
                        ((hi if country == "MEX" else (lo + hi) / 2), idx), xytext=(0, 8), textcoords="offset points",
                        ha="right" if country == "MEX" else "center", va="bottom", fontsize=6.2, color=COLORS["ink"])
        ax.set_yticks(yy, [service_labels[s] for s in service_order])
        ax.invert_yaxis()
        ax.set_ylim(len(service_order) - 0.45, -0.55)
        ax.set_title(country_name, loc="left", pad=6)

    legend = [
        Line2D([0], [0], color=COLORS["navy"], linewidth=2, marker="o", markersize=3.3,
               markerfacecolor=COLORS["navy"], label="Feasible coverage range"),
        Line2D([0], [0], linestyle="none", marker="D", markersize=4.2,
               color=COLORS["ochre"], label="Observed allocation"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.12),
               ncol=2, fontsize=6.3, handlelength=1.8)
    fig.text(0.5, 0.055, "Ranges include the observed allocation and eight state-constrained joint-composition constructions. Widths are lower bounds.",
             ha="center", fontsize=6.2, color=COLORS["slate"])
    fig.subplots_adjust(left=0.18, right=0.985, top=0.89, bottom=0.25, wspace=0.35)
    save_figure(fig, out, "Figure_4")
    state["observed_coverage_percent"] = state.apply(observed_value, axis=1)
    state["source_file"] = "upgrade_20261112/final_tables/stronger_constraint_widths.csv"
    save_source(out, "Figure_4", ("30-min all-population state and joint-composition constraints", state))


def figure5(d: dict[str, object], out: Path) -> None:
    services = ["chole", "cesarean", "appendectomy"]
    labels = {"chole": "Cholecystectomy", "cesarean": "Cesarean delivery", "appendectomy": "Appendectomy"}
    multi = d["multiservice"].set_index("service")
    prior = d["prior"].loc[d["prior"]["scheme"].eq("spatial_blocks")].set_index("service")
    temporal = d["temporal"].set_index("service")
    offsets = {"population + urban baseline": -0.22,
               "prior local-service baseline": 0.0,
               "2023 train → 2024 test": 0.22}
    colors = {"population + urban baseline": COLORS["navy"],
              "prior local-service baseline": COLORS["slate"],
              "2023 train → 2024 test": COLORS["ochre"]}
    markers = {"population + urban baseline": "o",
               "prior local-service baseline": "s",
               "2023 train → 2024 test": "D"}

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    style_axes(ax)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8, linestyle="--", zorder=1)
    for idx, service in enumerate(services):
        a = multi.loc[service]
        y = idx + offsets["population + urban baseline"]
        mean, lo, hi = float(a["mean"]), float(a["lo"]), float(a["hi"])
        ax.errorbar(mean, y, xerr=np.array([[mean - lo], [hi - mean]]), fmt=markers["population + urban baseline"],
                    color=colors["population + urban baseline"], ecolor=colors["population + urban baseline"],
                    elinewidth=1.15, capsize=2.2, markersize=4.1, zorder=3,
                    label="Population + urban baseline" if idx == 0 else None)

        b = prior.loc[service]
        y = idx + offsets["prior local-service baseline"]
        mean, lo, hi = float(b["mean_delta_auc"]), float(b["repartition_lo"]), float(b["repartition_hi"])
        ax.errorbar(mean, y, xerr=np.array([[mean - lo], [hi - mean]]), fmt=markers["prior local-service baseline"],
                    color=colors["prior local-service baseline"], ecolor=colors["prior local-service baseline"],
                    elinewidth=1.15, capsize=2.2, markersize=4.1, zorder=3,
                    label="Prior local-service baseline" if idx == 0 else None)
        if service == "cesarean":
            ax.annotate(f"{mean:+.5f}", (mean, y), xytext=(-2, 8), textcoords="offset points",
                        ha="right", va="bottom", fontsize=6.1, color=colors["prior local-service baseline"])

        t = temporal.loc[service]
        y = idx + offsets["2023 train → 2024 test"]
        val = float(t["activity_delta_auc"])
        ax.scatter(val, y, marker=markers["2023 train → 2024 test"], s=22,
                   color=colors["2023 train → 2024 test"], zorder=3,
                   label="2023 train → 2024 test" if idx == 0 else None)

    ax.set_yticks(np.arange(len(services)), [labels[s] for s in services])
    ax.invert_yaxis()
    ax.set_xlim(-0.0022, 0.0215)
    ax.set_xticks([-0.002, 0, 0.005, 0.010, 0.015, 0.020])
    ax.set_xlabel("Incremental AUC after adding procedure activity")
    ax.set_title("AUC increments across services and validation designs", loc="left", pad=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.23), ncol=3, fontsize=6.0, handlelength=1.4, labelspacing=0.4)
    fig.text(0.5, 0.06, "Spatial ranges show 2.5th–97.5th percentiles across 50 repartitions. Temporal values are point estimates.",
             ha="center", fontsize=6.3, color=COLORS["slate"])
    fig.subplots_adjust(left=0.18, right=0.985, top=0.88, bottom=0.30)
    save_figure(fig, out, "Figure_5")

    multi_source = d["multiservice"].copy()
    multi_source["baseline"] = "population + urban"
    multi_source["source_file"] = "upgrade_20261112/final_tables/multiservice_spatial_validation.csv"
    prior_source = d["prior"].loc[d["prior"]["scheme"].eq("spatial_blocks")].copy()
    prior_source["baseline"] = "prior local service"
    prior_source["source_file"] = "upgrade_20261112/final_tables/prior_service_adjusted_validation.csv"
    temporal_source = d["temporal"].copy()
    temporal_source["validation_design"] = "2023 train, 2024 test"
    temporal_source["source_file"] = "upgrade_20261112/final_tables/temporal_validation.csv"
    save_source(out, "Figure_5", ("population + urban baseline, 50 spatial repartitions", multi_source),
                ("prior local-service baseline, 50 spatial repartitions", prior_source),
                ("2023 train to 2024 test temporal holdout", temporal_source))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate five manuscript figures from existing source tables.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--upgrade-root", type=Path, default=DEFAULT_UPGRADE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    upgrade_root = args.upgrade_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    d = read_inputs(paths_for(project_root, upgrade_root))
    validate_sources(d)
    out.mkdir(parents=True, exist_ok=True)
    figure1(d, out)
    figure2(d, out)
    figure3(d, out)
    figure4(d, out)
    figure5(d, out)
    print(f"Created five main figures (PDF, SVG, 600-dpi PNG, LZW TIFF) and source CSVs in {out}")
    print("CERTIFIED_27 median_pp={:.9f} p90_pp={:.9f} max_COM_pp={:.9f} Spearman_rho={:.8f}".format(
        float(d["certified"]["width"].median() * 100),
        float(d["certified"]["width"].quantile(0.9) * 100),
        float(d["certified"].loc[d["certified"]["iso"].eq("COM"), "width"].iloc[0] * 100),
        float(d["certified"]["observed"].corr(d["certified"]["width"], method="spearman"))))
    print("COMPOSITION_MASKS", d["masks"][["nominal_mix_u", "actual_global_q", "coverage30"]].to_dict(orient="records"))
    print("STATE_WIDTH_LOWER_BOUNDS", d["strong"].loc[
        d["strong"]["threshold_minutes"].eq(30) & d["strong"]["population_group"].eq("all")
        & d["strong"]["folder"].isin(["brazil_state_constraints", "mexico_state_constraints"]),
        ["country", "service", "k", "n_candidates", "width_lower_bound_pp"]].to_dict(orient="records"))


if __name__ == "__main__":
    main()
