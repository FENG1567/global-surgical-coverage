#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create readable, source-linked supplementary figures from released figure tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "figures" / "supplement"
WIDTH_MM = 183.0
WIDTH_IN = WIDTH_MM / 25.4
PNG_RESOLUTION = 300
TIFF_DPI = 600
BLUE = "#326C8A"
ORANGE = "#C47A28"
GRAY = "#6B6B6B"
LIGHT_GRAY = "#B9B9B9"
COUNTRY_NAMES = {"BLZ": "Belize", "DJI": "Djibouti", "GNB": "Guinea-Bissau"}
SERVICE_NAMES = {"chole": "Cholecystectomy", "cesarean": "Cesarean delivery", "appendectomy": "Appendectomy"}
CAPTIONS_EN = {
    "Figure_S1": (
        "Fig. S1 | Spatial validation and Mexico scenario coverage. a, Mean incremental AUC under "
        "activity-augmented models for strict cholecystectomy in Brazil across three spatial evaluation "
        "schemes, using either the population + urban baseline or the prior-service baseline. Intervals "
        "are shown where 50 spatial-block repartitions are available; leave-region-out results with one "
        "evaluation are shown as points. b, Observed, random composition-matched and clustered minimum "
        "30-minute population coverage for cholecystectomy and cesarean delivery in Mexico. Horizontal "
        "spans show the observed range across designated joint-composition scenarios; the 'at least' "
        "label gives span width. These are scenario ranges, not confidence intervals."
    ),
    "Figure_S2": (
        "Fig. S2 | Information value of facility labels. Mean remaining coverage width as facility labels "
        "are revealed under count-only or count + exact-composition information and random or "
        "largest-catchment selection in Belize, Djibouti and Guinea-Bissau. Curves summarize 100 truth "
        "assignments. Shaded bands show interquartile ranges for count-only curves. The horizontal "
        "coordinate is the nominal target reveal fraction; its simulated count is rounded to an integer number of facilities in n_labels_revealed. Panel "
        "titles report selected/total facilities (k/n)."
    ),
    "Figure_S3": (
        "Fig. S3 | Brazil multiservice coverage scenarios and validation. a, 30-minute population "
        "coverage for observed, random joint-composition-matched and clustered-minimum scenarios across "
        "cholecystectomy, cesarean delivery and appendectomy. Horizontal spans show the range across "
        "designated observed/matched/joint-search scenarios; labels report constructed spread in "
        "percentage points, not an optimum or confidence interval. b, Incremental AUC for activity-based "
        "access over baseline for the three services. Points and intervals are reported summaries across "
        "50 repeated spatial-block repartitions."
    ),
    "Figure_S4": (
        "Fig. S4 | Coverage bounds under stronger constraints and validation dependence. a, 30-minute "
        "all-population coverage endpoints under exact state and joint-composition quotas for five "
        "country/service combinations; annotations show constructed width lower bounds, not optimal or "
        "uncertainty intervals. b, Incremental AUC for population + urban, prior-service-adjusted and "
        "temporal-holdout comparisons. Intervals are shown only for the first two comparisons, which "
        "supply repeated-repartition summaries; temporal holdout is shown as a point. The three series "
        "use distinct validation designs and are not pooled."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Directory containing released figure source CSV files.")
    parser.add_argument("--repo", type=Path, help="Repository root containing data/exact_cases/exhaustive_endpoints.csv.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory.")
    args = parser.parse_args()
    repo = args.repo.resolve() if args.repo else REPO_ROOT
    if args.data:
        data = args.data.resolve()
    else:
        candidates = [
            repo / "data" / "figure_sources",
            repo.parent / "public_upload" / "data" / "figure_sources",
            repo.parent / "repository" / "data" / "figure_sources",
        ]
        data = next((p.resolve() for p in candidates if p.is_dir()), candidates[0].resolve())
    args.data, args.repo, args.out = data, repo, args.out.resolve()
    return args


def read_table(data_dir: Path, name: str, expected_rows: int | None = None) -> pd.DataFrame:
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Required figure source is missing: {path}")
    frame = pd.read_csv(path)
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} data rows, found {len(frame)}")
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


def lookup_one(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {label}, found {len(selected)}")
    return selected.iloc[0]


def add_panel_letter(ax, letter: str) -> None:
    ax.text(-0.09, 1.04, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="left", clip_on=False)


def interval_point(ax, x: float, y: float, lo: float, hi: float, color: str, marker: str,
                   *, ms: float = 4.0, lw: float = 1.3) -> None:
    if not (lo <= x <= hi):
        raise ValueError(f"Point {x} falls outside supplied interval [{lo}, {hi}]")
    ax.errorbar(x, y, xerr=np.array([[x - lo], [hi - x]]), fmt=marker, color=color,
                ecolor=color, elinewidth=lw, capsize=2, capthick=lw, markersize=ms,
                markeredgewidth=0.5, zorder=3)


def set_figure_size(fig, height: float) -> None:
    fig.set_size_inches(WIDTH_IN, height, forward=True)


def export_figure(fig, out_dir: Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = map(float, fig.get_size_inches())
    if abs(width - WIDTH_IN) > 1e-8:
        raise ValueError(f"{name}: expected {WIDTH_IN:.6f} in width, got {width:.6f}")
    base = out_dir / name
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    tight_bbox_in = [float(tight_bbox.x0), float(tight_bbox.y0),
                     float(tight_bbox.x1), float(tight_bbox.y1)]
    if (tight_bbox.x0 < -0.005 or tight_bbox.y0 < -0.005
            or tight_bbox.x1 > width + 0.005 or tight_bbox.y1 > height + 0.005):
        raise ValueError(f"{name}: tight bounding box {tight_bbox_in} exceeds canvas {width} x {height} in")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=PNG_RESOLUTION, facecolor="white")
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        fig.savefig(tmp.name, format="png", dpi=TIFF_DPI, facecolor="white")
        with Image.open(tmp.name) as raster:
            raster.convert("RGB").save(base.with_suffix(".tiff"), compression="tiff_lzw", dpi=(TIFF_DPI, TIFF_DPI))
    with Image.open(base.with_suffix(".tiff")) as raster:
        if raster.mode != "RGB":
            raise ValueError(f"{name}: TIFF is {raster.mode}, expected RGB")
        if abs(raster.size[0] - round(WIDTH_IN * TIFF_DPI)) > 1:
            raise ValueError(f"{name}: TIFF width {raster.size[0]} px; expected {round(WIDTH_IN * TIFF_DPI)}")
        tiff_size, tiff_dpi = list(raster.size), [float(v) for v in raster.info.get("dpi", ())]
    with Image.open(base.with_suffix(".png")) as raster:
        png_size = list(raster.size)
    plt.close(fig)
    return {
        "width_mm": WIDTH_MM, "height_in": height, "tight_bbox_in": tight_bbox_in,
        "png_pixels": png_size,
        "tiff_pixels": tiff_size, "tiff_dpi": tiff_dpi, "tiff_mode": "RGB",
        "tiff_compression": "LZW",
        "files": {ext: {"path": f"{name}.{ext}", "bytes": base.with_suffix("." + ext).stat().st_size}
                  for ext in ("pdf", "png", "svg", "tiff")},
    }


def build_s1(data_dir: Path):
    validation = read_table(data_dir, "validation_panel_source.csv", 6)
    mexico = read_table(data_dir, "mexico_panel_source.csv", 51)
    require_columns(validation, ["scheme", "comparison", "mean_delta_auc", "repartition_lo",
                                 "repartition_hi", "positive_repartitions", "n_repartitions"], "validation_panel_source.csv")
    require_columns(mexico, ["scenario", "service", "all_30"], "mexico_panel_source.csv")
    fig, (ax, bx) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.22, 1.0]})
    set_figure_size(fig, 3.8)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.79, bottom=0.18, wspace=0.48)
    schemes = [("spatial_blocks", "Spatial blocks"), ("leave_region_out", "Leave-region-out"),
               ("leave_region_out_buffered", "Buffered holdout")]
    comparisons = [
        ("population_urban_activity minus population_urban", "Pop. + urban baseline", BLUE, "o"),
        ("prior_service_plus_activity minus prior_service_baseline", "Prior-service baseline", GRAY, "s"),
    ]
    panel_audit, ylabels = [], []
    index = 0
    for comparison, baseline_label, color, marker in comparisons:
        for scheme, scheme_label in schemes:
            row = lookup_one(validation, validation.scheme.eq(scheme) & validation.comparison.eq(comparison),
                             f"{scheme}/{comparison}")
            value, lo, hi = float(row.mean_delta_auc), float(row.repartition_lo), float(row.repartition_hi)
            if int(row.n_repartitions) > 1:
                interval_point(ax, value, index, lo, hi, color, marker)
            else:
                ax.scatter(value, index, s=25, color=color, marker=marker, edgecolor="white",
                           linewidth=0.5, zorder=3)
            ylabels.append(f"{scheme_label}\n{baseline_label}")
            panel_audit.append({
                "scheme": scheme, "comparison": comparison, "mean_delta_auc": value,
                "repartition_lo": lo, "repartition_hi": hi,
                "positive_repartitions": int(row.positive_repartitions),
                "n_repartitions": int(row.n_repartitions),
            })
            index += 1
    ax.axvline(0, color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    ax.set_xlim(-0.001, 0.018)
    ax.set_xticks([0, 0.005, 0.010, 0.015])
    ax.set_xlabel("Incremental AUC (activity model - baseline)")
    ax.set_yticks(range(6), ylabels)
    ax.invert_yaxis()
    ax.set_ylim(5.75, -0.75)
    ax.set_title("Brazil: strict cholecystectomy", loc="left", pad=7)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.55)
    ax.set_axisbelow(True)
    add_panel_letter(ax, "a")

    panel_b_audit = []
    styles = [("observed", "Observed", BLUE, "o"),
              ("random_jointmatched", "Random, fixed joint composition", GRAY, "s")]
    for i, service in enumerate(("chole", "cesarean")):
        subset = mexico.loc[mexico.service.eq(service)]
        eligible = subset.loc[subset.scenario.str.contains(r"observed|jointmatched|jointsearch", regex=True)]
        if subset.empty or eligible.empty:
            raise ValueError(f"Mexico source has no designated rows for {service}")
        lo, hi = float(eligible.all_30.min() * 100), float(eligible.all_30.max() * 100)
        bx.hlines(i, lo, hi, color=LIGHT_GRAY, linewidth=2.0, zorder=1)
        points = []
        for suffix, label, color, marker in styles:
            row = lookup_one(subset, subset.scenario.eq(f"{service}_{suffix}"), f"{service}/{suffix}")
            value = float(row.all_30 * 100)
            bx.scatter(value, i, s=30, color=color, marker=marker, edgecolor="white",
                        linewidth=0.5, zorder=3)
            points.append({"scenario": suffix, "label": label, "all_30_pct": value})
        clustered = eligible.loc[eligible.scenario.str.contains(r"cluster_jointmatched|jointsearch", regex=True)]
        if clustered.empty:
            raise ValueError(f"Mexico source has no clustered/joint-search rows for {service}")
        cluster_min = float(clustered.all_30.min() * 100)
        bx.scatter(cluster_min, i, s=31, color=ORANGE, marker="D", edgecolor="white",
                    linewidth=0.5, zorder=4)
        points.append({"scenario": "clustered minimum", "label": "Clustered minimum", "all_30_pct": cluster_min})
        spread = hi - lo
        bx.text(lo, i + 0.18, f"At least {spread:.2f} pp", fontsize=7.0, ha="left", va="top")
        panel_b_audit.append({"service": service, "range_min_pct": lo, "range_max_pct": hi,
                              "range_width_pp": spread, "points": points, "range_source_rows": int(len(eligible))})
    bx.set_yticks([0, 1], [SERVICE_NAMES["chole"], SERVICE_NAMES["cesarean"]])
    bx.set_ylim(1.55, -0.55)
    bx.set_xlim(65, 95)
    bx.set_xticks([65, 75, 85, 95])
    bx.set_xlabel("30-minute population coverage (%)")
    bx.set_title("Mexico: two services", loc="left", pad=7)
    bx.grid(axis="x", color="#E8E8E8", linewidth=0.55)
    bx.set_axisbelow(True)
    add_panel_letter(bx, "b")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=BLUE, markersize=4.5, label="Observed"),
        Line2D([0], [0], marker="s", linestyle="none", color=GRAY, markersize=4.5, label="Random, fixed joint composition"),
        Line2D([0], [0], marker="D", linestyle="none", color=ORANGE, markersize=4.5, label="Clustered minimum"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               bbox_transform=fig.transFigure, frameon=False, fontsize=7.0, ncol=3,
               columnspacing=1.2, handletextpad=0.5)
    return fig, {"a": panel_audit, "b": panel_b_audit}


def build_s2(data_dir: Path, repo: Path | None):
    sim = read_table(data_dir, "information_value_panel_source.csv", 6000)
    if repo is None:
        raise FileNotFoundError("Pass --repo for data/exact_cases/exhaustive_endpoints.csv.")
    endpoint_path = repo / "data" / "exact_cases" / "exhaustive_endpoints.csv"
    if not endpoint_path.is_file():
        raise FileNotFoundError(f"Required exact endpoint source is missing: {endpoint_path}")
    endpoints = pd.read_csv(endpoint_path)
    if len(endpoints) != 3:
        raise ValueError(f"Expected three endpoint rows, found {len(endpoints)}")
    require_columns(sim, ["iso", "truth_rep", "prior", "strategy", "fraction",
                          "n_labels_revealed", "remaining_width"], "information_value_panel_source.csv")
    require_columns(endpoints, ["iso", "n", "k"], "exhaustive_endpoints.csv")
    groups = sim.groupby(["iso", "prior", "strategy", "fraction"])
    if groups.truth_rep.nunique().ne(100).any():
        raise ValueError("Every Figure S2 curve point must retain all 100 truth assignments.")
    if groups.n_labels_revealed.nunique().ne(1).any():
        raise ValueError("Integer labels revealed must be constant within every curve point.")
    if endpoints.iso.nunique() != 3 or set(sim.iso.unique()) != set(endpoints.iso):
        raise ValueError("Simulation countries do not match endpoint rows.")
    fig, axes = plt.subplots(1, 3, sharey=True)
    set_figure_size(fig, 4.0)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.75, bottom=0.17, wspace=0.23)
    colors = {"random": BLUE, "largest_catchment": ORANGE}
    markers = {"random": "o", "largest_catchment": "s"}
    priors = [("count_only", "-", "count only"),
              ("count_and_exact_q", (0, (5, 2)), "count + exact composition")]
    panel_audit = {}
    for ax, iso, letter in zip(axes, ("BLZ", "DJI", "GNB"), ("a", "b", "c")):
        endpoint = lookup_one(endpoints, endpoints.iso.eq(iso), f"endpoint/{iso}")
        n, k = int(endpoint.n), int(endpoint.k)
        subset = sim.loc[sim.iso.eq(iso)]
        country_audit = {"selected_k": k, "total_n": n, "curves": []}
        for prior, linestyle, prior_label in priors:
            for strategy in ("random", "largest_catchment"):
                curve = subset.loc[subset.prior.eq(prior) & subset.strategy.eq(strategy)]
                points, xs, means, lows, highs = [], [], [], [], []
                for fraction, group in curve.groupby("fraction", sort=True):
                    n_labels = int(group.n_labels_revealed.iloc[0])
                    x = float(fraction) * 100.0
                    actual_reveal = n_labels / n * 100.0
                    mean = float(group.remaining_width.mean() * 100.0)
                    low, high = float(group.remaining_width.quantile(0.25) * 100.0), float(group.remaining_width.quantile(0.75) * 100.0)
                    xs.append(x)
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
                    points.append({
                        "target_fraction": float(fraction), "nominal_reveal_pct": x,
                        "n_labels_revealed": n_labels, "actual_reveal_pct": actual_reveal,
                        "mean_remaining_width_pp": mean,
                        "q25_pp": low if prior == "count_only" else None,
                        "q75_pp": high if prior == "count_only" else None,
                        "truth_assignments": int(group.truth_rep.nunique()),
                    })
                if len(xs) != 5 or np.any(np.diff(xs) <= 0):
                    raise ValueError(f"{iso}/{prior}/{strategy}: expected five ordered curve points.")
                ax.plot(xs, means, color=colors[strategy], linestyle=linestyle, linewidth=1.45,
                        marker=markers[strategy], markersize=3.1, markeredgewidth=0.45)
                if prior == "count_only":
                    ax.fill_between(xs, lows, highs, color=colors[strategy], alpha=0.12, linewidth=0)
                country_audit["curves"].append({
                    "prior": prior, "prior_label": prior_label, "strategy": strategy,
                    "points": points, "iqr_shown": prior == "count_only",
                })
        ax.set_title(f"{COUNTRY_NAMES[iso]}\nSelected / total = {k}/{n}", pad=5)
        ax.set_xlim(0, 65)
        ax.set_xticks([0, 10, 20, 40, 60])
        ax.set_xlabel("Labels revealed (nominal %)")
        ax.set_ylim(0, 21)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.55)
        ax.set_axisbelow(True)
        add_panel_letter(ax, letter)
        panel_audit[iso] = country_audit
    axes[0].set_ylabel("Mean remaining coverage width (pp)")
    fig.suptitle("Information value of facility labels", y=0.86, fontsize=8.5, fontweight="bold")
    handles = []
    for prior, linestyle, prior_label in priors:
        for strategy in ("random", "largest_catchment"):
            strategy_label = "Random" if strategy == "random" else "Largest catchment"
            handles.append(Line2D([0], [0], color=colors[strategy], linestyle=linestyle,
                                  marker=markers[strategy], markersize=3.5, linewidth=1.4,
                                  label=f"{strategy_label} | {prior_label}"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               bbox_transform=fig.transFigure, frameon=False, fontsize=7.0, ncol=2,
               columnspacing=1.4, handlelength=4.0, handletextpad=0.5, labelspacing=0.25)
    return fig, panel_audit


def scenario_range_points(frame: pd.DataFrame, service: str):
    subset = frame.loc[frame.service.eq(service)]
    eligible = subset.loc[subset.scenario.str.contains(r"observed|jointmatched|jointsearch", regex=True)]
    if eligible.empty:
        raise ValueError(f"{service}: no designated scenario rows.")
    observed = lookup_one(subset, subset.scenario.eq(f"{service}_observed"), f"{service}/observed")
    random = lookup_one(subset, subset.scenario.eq(f"{service}_random_jointmatched"), f"{service}/random_jointmatched")
    clustered = eligible.loc[eligible.scenario.str.contains(r"cluster_jointmatched|jointsearch", regex=True)]
    if clustered.empty:
        raise ValueError(f"{service}: no clustered/joint-search rows.")
    values = {
        "observed": float(observed.all_30 * 100),
        "random_jointmatched": float(random.all_30 * 100),
        "clustered_minimum": float(clustered.all_30.min() * 100),
    }
    return eligible, values


def build_s3(data_dir: Path):
    brazil = read_table(data_dir, "brazil_panel_source.csv", 64)
    validation = read_table(data_dir, "multiservice_validation_panel_source.csv", 9)
    require_columns(brazil, ["scenario", "service", "all_30"], "brazil_panel_source.csv")
    require_columns(validation, ["service", "comparison", "mean", "lo", "hi", "n_repartitions"],
                    "multiservice_validation_panel_source.csv")
    fig, (ax, bx) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.0]})
    set_figure_size(fig, 3.85)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.79, bottom=0.17, wspace=0.38)
    styles = {
        "observed": ("Observed", BLUE, "o"),
        "random_jointmatched": ("Random, fixed joint composition", GRAY, "s"),
        "clustered_minimum": ("Clustered minimum", ORANGE, "D"),
    }
    panel_audit, validation_audit = [], []
    services = ("chole", "cesarean", "appendectomy")
    for i, service in enumerate(services):
        eligible, values = scenario_range_points(brazil, service)
        lo, hi = float(eligible.all_30.min() * 100), float(eligible.all_30.max() * 100)
        ax.hlines(i, lo, hi, color=LIGHT_GRAY, linewidth=2.0, zorder=1)
        for key, (label, color, marker) in styles.items():
            ax.scatter(values[key], i, s=29, color=color, marker=marker, edgecolor="white",
                       linewidth=0.5, zorder=3)
        spread_y = i + (0.18 if i % 2 == 0 else -0.18)
        ax.text((lo + hi) / 2.0, spread_y, f"Constructed spread: {hi - lo:.2f} pp",
                fontsize=7.0, ha="center", va="top" if i % 2 == 0 else "bottom")
        panel_audit.append({"service": service, "range_min_pct": lo, "range_max_pct": hi,
                            "constructed_spread_pp": hi - lo, "points_pct": values,
                            "range_source_rows": int(len(eligible))})
        row = lookup_one(validation, validation.service.eq(service) &
                         validation.comparison.eq("+activity - baseline"), f"validation/{service}")
        mean, low, high = float(row["mean"]), float(row.lo), float(row.hi)
        interval_point(bx, mean, i, low, high, BLUE, "o")
        validation_audit.append({"service": service, "mean": mean, "lo": low, "hi": high,
                                 "n_repartitions": int(row.n_repartitions)})
    for panel in (ax, bx):
        panel.set_yticks(range(3), [SERVICE_NAMES[s] for s in services])
        panel.set_ylim(2.65, -0.65)
        panel.grid(axis="x", color="#E8E8E8", linewidth=0.55)
        panel.set_axisbelow(True)
    ax.set_xlim(30, 90)
    ax.set_xticks([30, 40, 50, 60, 70, 80, 90])
    ax.set_xlabel("30-minute population coverage (%)")
    ax.set_title("Brazil: national joint-composition counts", loc="left", pad=7)
    bx.axvline(0, color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    bx.set_xlim(-0.002, 0.021)
    bx.set_xticks([0, 0.005, 0.010, 0.015, 0.020])
    bx.set_xlabel("Incremental AUC (activity - baseline)")
    bx.set_title("Repeated spatial-block validation", loc="left", pad=7)
    add_panel_letter(ax, "a")
    add_panel_letter(bx, "b")
    handles = [Line2D([0], [0], marker=marker, linestyle="none", color=color,
                      markersize=4.5, label=label) for label, color, marker in styles.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               bbox_transform=fig.transFigure, frameon=False, fontsize=7.0, ncol=3,
               columnspacing=1.2, handletextpad=0.5)
    return fig, {"a": panel_audit, "b": validation_audit}


def build_s4(data_dir: Path):
    state = read_table(data_dir, "state_constraint_panel_source.csv", 5)
    multiservice = read_table(data_dir, "multiservice_validation_panel_source.csv", 9)
    prior = read_table(data_dir, "strong_baseline_panel_source.csv", 9)
    temporal = read_table(data_dir, "temporal_validation.csv", 3)
    require_columns(state, ["country", "service", "threshold_minutes", "population_group",
                            "low_percent", "high_percent", "width_lower_bound_pp"], "state_constraint_panel_source.csv")
    require_columns(multiservice, ["service", "comparison", "mean", "lo", "hi"], "multiservice_validation_panel_source.csv")
    require_columns(prior, ["service", "scheme", "comparison", "mean_delta_auc",
                            "repartition_lo", "repartition_hi"], "strong_baseline_panel_source.csv")
    require_columns(temporal, ["service", "activity_delta_auc"], "temporal_validation.csv")
    if not state.threshold_minutes.eq(30).all() or not state.population_group.eq("all").all():
        raise ValueError("State-constraint source must contain 30-minute all-population bounds.")
    fig, (ax, bx) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.05]})
    set_figure_size(fig, 3.85)
    fig.subplots_adjust(left=0.23, right=0.99, top=0.79, bottom=0.17, wspace=0.38)
    order = [("BRA", "chole"), ("BRA", "cesarean"), ("BRA", "appendectomy"),
             ("MEX", "chole"), ("MEX", "cesarean")]
    bounds_audit = []
    for i, (country, service) in enumerate(order):
        row = lookup_one(state, state.country.eq(country) & state.service.eq(service), f"bounds/{country}/{service}")
        low, high, width = float(row.low_percent), float(row.high_percent), float(row.width_lower_bound_pp)
        ax.hlines(i, low, high, color=LIGHT_GRAY, linewidth=2.0, zorder=1)
        ax.scatter(low, i, color=ORANGE, marker="o", s=26, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(high, i, color=BLUE, marker="s", s=26, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(low - 0.8, i, f"{width:.2f}", ha="right", va="center", fontsize=7.0)
        bounds_audit.append({"country": country, "service": service, "low_percent": low,
                             "high_percent": high, "width_lower_bound_pp": width})
    ax.set_yticks(range(5), [f"{SERVICE_NAMES[service]}\n{country}" for country, service in order])
    ax.set_ylim(4.65, -0.65)
    ax.set_xlim(40, 95)
    ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_xlabel("30-minute population coverage (%)")
    ax.set_title("Constructed coverage bounds", loc="left", pad=7)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.55)
    ax.set_axisbelow(True)
    add_panel_letter(ax, "a")

    colors = {"population_urban": BLUE, "prior_service": GRAY, "temporal": ORANGE}
    markers = {"population_urban": "o", "prior_service": "s", "temporal": "D"}
    labels = {"population_urban": "Population + urban", "prior_service": "Prior-service adjusted",
              "temporal": "Temporal holdout (point)"}
    services = ("chole", "cesarean", "appendectomy")
    comparisons_audit = []
    for i, service in enumerate(services):
        pop = lookup_one(multiservice, multiservice.service.eq(service) &
                         multiservice.comparison.eq("+activity - baseline"), f"population+urban/{service}")
        prior_row = lookup_one(prior, prior.service.eq(service) & prior.scheme.eq("spatial_blocks") &
                               prior.comparison.eq("prior_service_plus_activity minus prior_service_baseline"),
                               f"prior service/{service}")
        time = lookup_one(temporal, temporal.service.eq(service), f"temporal/{service}")
        entries = [
            ("population_urban", float(pop["mean"]), float(pop.lo), float(pop.hi), -0.18),
            ("prior_service", float(prior_row.mean_delta_auc), float(prior_row.repartition_lo),
             float(prior_row.repartition_hi), 0.0),
            ("temporal", float(time.activity_delta_auc), None, None, 0.18),
        ]
        service_audit = {"service": service, "comparisons": []}
        for key, value, low, high, offset in entries:
            y = i + offset
            if low is not None:
                interval_point(bx, value, y, low, high, colors[key], markers[key], ms=3.8, lw=1.2)
            else:
                bx.scatter(value, y, color=colors[key], marker=markers[key], s=27,
                           edgecolor="white", linewidth=0.5, zorder=3)
            record = {"series": key, "value": value}
            if low is not None:
                record.update({"lo": low, "hi": high})
            service_audit["comparisons"].append(record)
        comparisons_audit.append(service_audit)
    bx.axvline(0, color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    bx.set_yticks(range(3), [SERVICE_NAMES[s] for s in services])
    bx.set_ylim(2.72, -0.72)
    bx.set_xlim(-0.0025, 0.021)
    bx.set_xticks([0, 0.005, 0.010, 0.015, 0.020])
    bx.set_xlabel("Incremental AUC (activity - comparison baseline)")
    bx.set_title("Three validation designs", loc="left", pad=7)
    bx.grid(axis="x", color="#E8E8E8", linewidth=0.55)
    bx.set_axisbelow(True)
    add_panel_letter(bx, "b")
    endpoint_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=ORANGE, markersize=4.5, label="Lower endpoint"),
        Line2D([0], [0], marker="s", linestyle="none", color=BLUE, markersize=4.5, label="Upper endpoint"),
    ]
    ax.legend(handles=endpoint_handles, loc="upper center", bbox_to_anchor=(0.29, 0.995),
              bbox_transform=fig.transFigure, frameon=False, fontsize=7.0, ncol=2,
              columnspacing=0.8, handletextpad=0.35)
    validation_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=BLUE, markersize=4.5, label=labels["population_urban"]),
        Line2D([0], [0], marker="s", linestyle="none", color=GRAY, markersize=4.5, label=labels["prior_service"]),
        Line2D([0], [0], marker="D", linestyle="none", color=ORANGE, markersize=4.5, label=labels["temporal"]),
    ]
    bx.legend(handles=validation_handles, loc="upper center", bbox_to_anchor=(0.76, 0.995),
              bbox_transform=fig.transFigure, frameon=False, fontsize=7.0, ncol=2,
              columnspacing=0.7, handletextpad=0.35, handlelength=1.2, labelspacing=0.25)
    return fig, {"a": bounds_audit, "b": comparisons_audit}


def source_metadata(data_dir: Path, repo: Path | None) -> dict:
    names = ["validation_panel_source.csv", "mexico_panel_source.csv",
             "information_value_panel_source.csv", "brazil_panel_source.csv",
             "multiservice_validation_panel_source.csv", "state_constraint_panel_source.csv",
             "strong_baseline_panel_source.csv", "temporal_validation.csv"]
    if repo:
        names.append("data/exact_cases/exhaustive_endpoints.csv")
    result = {}
    for name in names:
        path = repo / name if name.startswith("data/") and repo else data_dir / name
        payload = path.read_bytes()
        with path.open("rb") as stream:
            rows = sum(1 for _ in stream) - 1
        result[name] = {"data_rows": rows, "sha256": hashlib.sha256(payload).hexdigest()}
    return result


def main():
    args = parse_args()
    if not args.data.is_dir():
        raise FileNotFoundError(f"Figure source directory not found: {args.data}")
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.titleweight": "normal",
        "axes.labelsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })
    builders = [
        ("Figure_S1", lambda: build_s1(args.data)),
        ("Figure_S2", lambda: build_s2(args.data, args.repo)),
        ("Figure_S3", lambda: build_s3(args.data)),
        ("Figure_S4", lambda: build_s4(args.data)),
    ]
    audit = {
        "backend": "Python/matplotlib", "figure_width_mm": WIDTH_MM,
        "source_files": source_metadata(args.data, args.repo), "figures": {},
    }
    for name, builder in builders:
        fig, panel_audit = builder()
        audit["figures"][name] = {
            "panel_values": panel_audit, "exports": export_figure(fig, args.out, name),
        }
        print(f"EXPORTED {name}", flush=True)
    (args.out / "Figure_legends_en.txt").write_text(
        "\n\n".join(f"{name}: {caption}" for name, caption in CAPTIONS_EN.items()) + "\n",
        encoding="utf-8",
    )
    (args.out / "Figure_data_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"SUPPLEMENT_FIGURES_EXPORTED {args.out}", flush=True)


if __name__ == "__main__":
    main()
