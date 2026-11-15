#!/usr/bin/env python3
"""Pure-layout pass for main Figures 4–6; source data remain read-only."""
from __future__ import annotations
import argparse
import json
import itertools
from pathlib import Path
import sys

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_core_main as core
import layout_style as shared
import matplotlib.pyplot as plt
from matplotlib.text import Text


def legend_parts(legend):
    return legend.get_lines(), [t.get_text() for t in legend.get_texts()], legend.get_title().get_text()


def replace_fig_legend(fig, index, *, anchor, ncol, title=None, fontsize=8.5, columnspacing=0.8, handletextpad=0.35):
    old = fig.legends[index]
    handles, labels, old_title = legend_parts(old)
    old.remove()
    return fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=anchor,
                      ncol=ncol, fontsize=fontsize, columnspacing=columnspacing,
                      handletextpad=handletextpad, title=old_title if title is None else title,
                      title_fontsize=8.5 if (old_title if title is None else title) else None)


def replace_axis_legend(ax, *, anchor, ncol=1, fontsize=8.5, title=None):
    old = ax.get_legend()
    if old is None:
        return None
    handles, labels, old_title = legend_parts(old)
    old.remove()
    return ax.figure.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=anchor,
                            ncol=ncol, fontsize=fontsize, columnspacing=0.8,
                            handletextpad=0.35, title=old_title if title is None else title,
                            title_fontsize=8.5 if (old_title if title is None else title) else None)


def common_panel_labels(fig, axes):
    shared.remove_old_panel_labels(fig)
    for ax, letter in zip(axes, 'ABCD'):
        x0, y0, width, height = ax.get_position().bounds
        shared.label(fig, letter, max(0.005, x0 - 0.085), min(0.985, y0 + height + 0.018))


def overlap_boxes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bbox = text.get_window_extent(renderer)
        if bbox.width > 0 and bbox.height > 0:
            visible.append((text.get_text(), bbox))
    pairs = []
    for (text_a, a), (text_b, b) in itertools.combinations(visible, 2):
        dx = min(a.x1, b.x1) - max(a.x0, b.x0)
        dy = min(a.y1, b.y1) - max(a.y0, b.y0)
        if dx > 1.0 and dy > 1.0:
            fraction = dx * dy / min(a.width * a.height, b.width * b.height)
            if fraction > 0.12:
                pairs.append({'a': text_a, 'a_bbox': [round(v, 2) for v in a.extents],
                              'b': text_b, 'b_bbox': [round(v, 2) for v in b.extents],
                              'fraction': round(fraction, 3)})
    return pairs


def layout_4(fig):
    shared.bold_all(fig)
    axes = fig.axes
    assert len(axes) == 2
    # The 8 row categories are identical at 30 and 120 minutes, so retain one set.
    axes[1].tick_params(axis='y', labelleft=False)
    fig.subplots_adjust(left=0.315, right=0.960, top=0.900, bottom=0.205, wspace=0.26)
    for ax in axes:
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, 104)
    # Keep in-range ticks only; auto-locator edge ticks extended into the gutter.
    axes[0].set_xticks([40, 60, 80, 100])
    axes[1].set_xticks([80, 90, 100])
    legend = fig.legends[0]
    handles, labels, title = legend_parts(legend)
    legend.remove()
    fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.64, 0.078),
               ncol=3, fontsize=8.5, columnspacing=0.9, handletextpad=0.4)
    common_panel_labels(fig, axes)


def layout_5(fig):
    shared.bold_all(fig)
    axes = fig.axes
    assert len(axes) == 4
    ax_a, ax_b, ax_c, ax_d = axes
    # All three panels use the same service order; keep the readable y labels once.
    ax_b.tick_params(axis='y', labelleft=False)
    ax_d.tick_params(axis='y', labelleft=False)

    # Move all three independent legends into a dedicated two-row strip below the plots.
    replace_fig_legend(fig, 0, anchor=(0.50, 0.132), ncol=3, fontsize=8.5,
                       columnspacing=0.9, handletextpad=0.4,
                       title='Baseline in panels A and B')
    replace_axis_legend(ax_c, anchor=(0.31, 0.055), ncol=3, fontsize=8.5, title='Service')
    replace_axis_legend(ax_d, anchor=(0.77, 0.055), ncol=2, fontsize=8.5, title='2024 holdout')

    fig.subplots_adjust(left=0.175, right=0.950, top=0.885, bottom=0.235,
                        wspace=0.32, hspace=0.47)
    # Keep annotation labels clear of the right edge while leaving the data values intact.
    x0, x1 = ax_d.get_xlim()
    ax_d.set_xlim(x0, x1 + 0.040)
    common_panel_labels(fig, axes)


def layout_6(fig):
    shared.bold_all(fig)
    axes = fig.axes
    assert len(axes) == 4
    # The six candidate-set/rule categories are shared by panels A and B.
    axes[1].tick_params(axis='y', labelleft=False)
    fig.subplots_adjust(left=0.285, right=0.945, top=0.900, bottom=0.235,
                        wspace=0.18, hspace=0.52)
    legend = fig.legends[0]
    handles, labels, title = legend_parts(legend)
    legend.remove()
    fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.50, 0.090),
               ncol=3, fontsize=8.5, columnspacing=0.9, handletextpad=0.4)
    common_panel_labels(fig, axes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=P / 'main_4_6')
    parser.add_argument('--figures', nargs='+', choices=['4', '5', '6'], default=['4', '5', '6'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = core.read_sources(args.data)
    figures = {}
    baselines = {}

    def capture(fig, out, name, height_mm):
        figures[name] = fig
        # Capture the artist-coordinate signature immediately after the source plot function.
        baselines[name] = shared.signature(fig)

    core.savefig = capture
    source_metrics = {}
    draw_functions = {'4': core.figure4, '5': core.figure5, '6': core.figure6}
    selected = [f'Figure_{i}' for i in args.figures]
    for number in args.figures:
        source_metrics[f'Figure_{number}'] = draw_functions[number](data, args.out)

    layouts = {'Figure_4': (layout_4, 155), 'Figure_5': (layout_5, 190), 'Figure_6': (layout_6, 182)}
    audit_records = {}
    for name in selected:
        fig = figures[name]
        layout_fn, height_mm = layouts[name]
        fig.set_size_inches(shared.WIDTH_MM * shared.MM, height_mm * shared.MM, forward=True)
        layout_fn(fig)
        bbox_overlaps = overlap_boxes(fig)
        audit = shared.export(fig, args.out, name, height_mm, baseline_signature=baselines[name])
        audit_records[name] = {'baseline_signature': baselines[name], 'bbox_overlap_pairs': bbox_overlaps, 'layout_audit': audit}

    metrics_path = args.out / 'layout_metrics.json'
    existing = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}
    merged_sources = existing.get('source_metrics', {})
    merged_sources.update(source_metrics)
    merged_figures = existing.get('figures', {})
    merged_figures.update(audit_records)
    metrics_path.write_text(json.dumps(
        {'status': 'pass', 'source_metrics': merged_sources, 'figures': merged_figures},
        indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': 'pass', 'output': str(args.out),
                      'baseline_signatures': baselines,
                      'audits': {n: {'outside': len(audit_records[n]['layout_audit']['outside']),
                                     'text_overlaps': len(audit_records[n]['layout_audit']['overlaps']),
                                     'minimum_fontsize': audit_records[n]['layout_audit']['minimum_fontsize']}
                                 for n in audit_records}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
