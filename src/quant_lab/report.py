"""Figures (PNG) and a self-contained HTML report from one or more result files.

Charting rules: one y-axis per chart, small multiples per model, color only encodes the
execution provider (2 validated hues), variants are direct-labelled, values as text ink.
"""

from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path
from typing import Any

VARIANT_LABELS = {
    "fp32": "FP32",
    "fp16": "FP16",
    "int8-dynamic": "INT8 dyn",
    "int8-static": "INT8 static",
    "w8a16-static": "W8A16",
}
EP_COLORS = {
    "cpu": "#2a78d6",
    "coreml": "#eb6834",
    "qnn": "#1baf7a",
    "xnnpack": "#1baf7a",
    "nnapi": "#eda100",
    "cuda": "#6250d6",
}
EP_MARKERS = {"cpu": "o", "coreml": "s", "qnn": "^", "xnnpack": "^", "nnapi": "D", "cuda": "v"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(p).read_text()) for p in paths]


def ok_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in run["results"] if r.get("status") == "ok"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _place_labels(fig, ax, pts: list[dict[str, Any]]) -> None:
    """Direct labels: near-coincident points of the same EP share one label, remaining
    collisions are resolved by nudging vertically with a thin leader line."""
    fig.canvas.draw()
    pix = {id(r): ax.transData.transform((r["latency_ms"]["p50"], 100 * r["top1"])) for r in pts}
    clusters: list[list[dict[str, Any]]] = []
    for r in sorted(pts, key=lambda r: (r["provider"], r["latency_ms"]["p50"])):
        for c in clusters:
            x0, y0 = pix[id(c[0])]
            x, y = pix[id(r)]
            if c[0]["provider"] == r["provider"] and abs(x - x0) < 14 and abs(y - y0) < 14:
                c.append(r)
                break
        else:
            clusters.append([r])
    placed: list[tuple[float, float, float]] = []  # x, y, width (pixels)
    for c in sorted(clusters, key=lambda c: -max(r["top1"] for r in c)):
        text = " · ".join(
            VARIANT_LABELS[r["variant"]]
            for r in sorted(c, key=lambda r: list(VARIANT_LABELS).index(r["variant"]))
        )
        x, y = pix[id(c[0])]
        width = 5.2 * len(text)
        dy = 0.0
        for step in (0, 12, -12, 24, -24, 36, -36):
            dy = float(step)
            if all(
                abs((y + dy) - py) > 12 or x + 8 > px + pw or x + 8 + width < px
                for px, py, pw in placed
            ):
                break
        placed.append((x + 8, y + dy, width))
        r0 = c[0]
        ax.annotate(
            text,
            (r0["latency_ms"]["p50"], 100 * r0["top1"]),
            textcoords="offset points",
            xytext=(6, dy * 72 / fig.dpi),
            fontsize=7,
            color=INK2,
            va="center",
            arrowprops=None if dy == 0 else {"arrowstyle": "-", "color": GRID, "lw": 0.8},
        )


def figures(run: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ok_rows(run)
    models = list(dict.fromkeys(r["model"] for r in rows))
    eps = list(dict.fromkeys(r["provider"] for r in rows))
    variants = [v for v in VARIANT_LABELS if any(r["variant"] == v for r in rows)]
    paths = []
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK2,
            "text.color": INK,
        }
    )

    # 1) Accuracy vs latency trade-off, one panel per model.
    from matplotlib.ticker import FuncFormatter, LogLocator

    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.9), squeeze=False)
    handles = {}
    for ax, model in zip(axes[0], models, strict=True):
        _style(ax)
        pts_all = []
        for ep in eps:
            pts = [r for r in rows if r["model"] == model and r["provider"] == ep]
            label = {"cpu": "CPU", "coreml": "CoreML"}.get(ep, ep.upper())
            h = ax.scatter(
                [r["latency_ms"]["p50"] for r in pts],
                [100 * r["top1"] for r in pts],
                s=46,
                color=EP_COLORS.get(ep, INK2),
                marker=EP_MARKERS.get(ep, "o"),
                edgecolor=SURFACE,
                linewidth=1.5,
                label=label,
                zorder=3,
            )
            handles[label] = h
            pts_all += pts
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ys = [100 * r["top1"] for r in pts_all]
        pad = max(0.3, (max(ys) - min(ys)) * 0.12)
        ax.set_ylim(min(ys) - pad, max(ys) + pad * 1.6)
        ax.set_title(model, loc="left")
        ax.set_xlabel("latency p50, batch 1 (ms, log scale)")
        ax.set_ylabel("ImageNet top-1 (%)")
        _place_labels(fig, ax, pts_all)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper right",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Accuracy vs latency: upper-left is better",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths.append(out_dir / "tradeoff.png")
    fig.savefig(paths[-1], dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2) Accuracy change vs FP32 (CPU), percentage points.
    fig, axes = plt.subplots(
        1, len(models), figsize=(4.2 * len(models), 2.8), squeeze=False, sharey=True
    )
    for ax, model in zip(axes[0], models, strict=True):
        _style(ax)
        base = next(
            (
                r
                for r in rows
                if r["model"] == model and r["variant"] == "fp32" and r["provider"] == "cpu"
            ),
            None,
        )
        if base is None:
            continue
        vals = [
            (VARIANT_LABELS[v], 100 * (r["top1"] - base["top1"]))
            for v in variants
            if v != "fp32"
            for r in rows
            if r["model"] == model and r["variant"] == v and r["provider"] == "cpu"
        ]
        ys = range(len(vals))
        ax.barh(list(ys), [v for _, v in vals], color=EP_COLORS["cpu"], height=0.6)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(list(ys), [n for n, _ in vals])
        ax.invert_yaxis()
        for y, (_, v) in zip(ys, vals, strict=True):
            ax.annotate(
                f"{v:+.1f} pp",
                (v, y),
                xytext=(4 if v >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=7,
                color=INK,
            )
        lo = min(-2.0, min(v for _, v in vals) * 1.25)
        ax.set_xlim(lo, max(2.0, max(v for _, v in vals) * 1.25 + 1))
        ax.set_title(model, loc="left")
        ax.set_xlabel("top-1 change vs FP32 (pp, CPU)")
    fig.tight_layout()
    paths.append(out_dir / "accuracy_drop.png")
    fig.savefig(paths[-1], dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 3) Latency per variant, CPU vs CoreML (grouped bars).
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.0), squeeze=False)
    h = 0.8 / max(1, len(eps))
    for ax, model in zip(axes[0], models, strict=True):
        _style(ax)
        for j, ep in enumerate(eps):
            vals = []
            for v in variants:
                r = next(
                    (
                        x
                        for x in rows
                        if x["model"] == model and x["variant"] == v and x["provider"] == ep
                    ),
                    None,
                )
                vals.append(r["latency_ms"]["p50"] if r else 0)
            ys = [i + j * h for i in range(len(variants))]
            ax.barh(
                ys,
                vals,
                height=h * 0.9,
                color=EP_COLORS.get(ep, INK2),
                label={"cpu": "CPU", "coreml": "CoreML"}.get(ep, ep.upper()),
            )
        ax.set_yticks(
            [i + h * (len(eps) - 1) / 2 for i in range(len(variants))],
            [VARIANT_LABELS[v] for v in variants],
        )
        ax.invert_yaxis()
        ax.set_title(model, loc="left")
        ax.set_xlabel("latency p50, batch 1 (ms)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(out_dir / "latency.png")
    fig.savefig(paths[-1], dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 4) Model size per variant.
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 2.6), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        _style(ax)
        vals = []
        for v in variants:
            r = next((x for x in rows if x["model"] == model and x["variant"] == v), None)
            vals.append(r["size_mb"] if r else 0)
        ax.barh(range(len(variants)), vals, color=EP_COLORS["cpu"], height=0.6)
        ax.set_yticks(range(len(variants)), [VARIANT_LABELS[v] for v in variants])
        ax.invert_yaxis()
        for y, v in enumerate(vals):
            ax.annotate(
                f"{v:.1f} MB",
                (v, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=INK,
            )
        ax.set_xlim(0, max(vals) * 1.3)
        ax.set_title(model, loc="left")
        ax.set_xlabel("ONNX file size (MB)")
    fig.tight_layout()
    paths.append(out_dir / "size.png")
    fig.savefig(paths[-1], dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return paths


def _cov(r: dict[str, Any]) -> str:
    c = r.get("ep_coverage")
    if not c:
        return "all (CPU)" if r["provider"] == "cpu" else "n/a"
    return f"{c['nodes_on_ep']}/{c['nodes_total']} nodes, {c['partitions']} part."


def table_rows(run: dict[str, Any]) -> list[list[str]]:
    out = []
    for r in run["results"]:
        if r.get("status") != "ok":
            out.append(
                [
                    r["model"],
                    VARIANT_LABELS.get(r["variant"], r["variant"]),
                    r["provider"],
                    r.get("status", "?"),
                    "",
                    "",
                    "",
                    "",
                    "",
                    escape(r.get("error", ""))[:80],
                ]
            )
            continue
        out.append(
            [
                r["model"],
                VARIANT_LABELS[r["variant"]],
                r["provider"],
                f"{100 * r['top1']:.2f}%",
                f"{100 * r['top1_10way']:.2f}%",
                f"{100 * r.get('agreement_with_fp32', float('nan')):.2f}%",
                f"{r['latency_ms']['p50']:.2f}",
                f"{r['latency_ms']['p99']:.2f}",
                f"{r['size_mb']:.1f}",
                _cov(r),
            ]
        )
    return out


HEAD = [
    "Model",
    "Variant",
    "EP",
    "Top-1 (1000-way)",
    "Top-1 (10-way)",
    "Agree w/ FP32",
    "p50 ms",
    "p99 ms",
    "MB",
    "EP coverage",
]

CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;
--muted:#c3c2b7;--line:#33332f;--card:#222220}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#33332f;--card:#222220}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px}h1{margin:0 0 4px}.muted{color:var(--muted)}
figure{margin:18px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:10px;
padding:8px;overflow-x:auto}figure img{width:100%;min-width:640px;height:auto;display:block}
figcaption{color:#52514e;font-size:12px;padding:4px 6px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;
font-size:13px}th{color:var(--muted)}
"""


def html(runs: list[dict[str, Any]], pngs: list[Path], ablation: dict | None) -> str:
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Quant Lab Report</title><style>{CSS}</style></head><body><main>",
        "<h1>Quant Lab</h1><p class='muted'>FP32 vs FP16 vs INT8 (dynamic / static) vs "
        "W8A16 on Imagenette (ImageNet top-1, batch 1).</p>",
    ]
    for run in runs:
        d = run.get("device", {})
        parts.append(
            f"<h2>{escape(str(run.get('label')))}</h2><p class='muted'>"
            f"{escape(str(d.get('os')))} {escape(str(d.get('os_version', '')))} · "
            f"onnxruntime {escape(str(d.get('onnxruntime')))} · {run['n_images']} images · "
            f"{escape(run['created'])}</p>"
        )
    captions = {
        "tradeoff": "Accuracy vs latency per model (color = execution provider).",
        "accuracy_drop": "Top-1 change vs FP32 on CPU (percentage points).",
        "latency": "Batch-1 latency per variant and execution provider.",
        "size": "ONNX file size per variant.",
    }
    for p in pngs:
        b64 = base64.b64encode(p.read_bytes()).decode()
        parts.append(
            f"<figure><img alt='{escape(captions.get(p.stem, p.stem))}' "
            f"src='data:image/png;base64,{b64}'><figcaption>"
            f"{escape(captions.get(p.stem, p.stem))}</figcaption></figure>"
        )
    for run in runs:
        parts.append(
            f"<h3>All measurements · {escape(str(run.get('label')))}</h3><div class='scroll'>"
            "<table><tr>" + "".join(f"<th>{h}</th>" for h in HEAD) + "</tr>"
        )
        for row in table_rows(run):
            parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        parts.append("</table></div>")
    if ablation:
        parts.append(
            "<h3>INT8 recipe ablation (dev set from the training split)</h3>"
            "<div class='scroll'><table><tr><th>Model</th><th>Recipe</th><th>Top-1</th>"
            "<th>Agree w/ FP32</th></tr>"
        )
        for model, res in ablation["results"].items():
            for recipe, v in res.items():
                parts.append(
                    f"<tr><td>{model}</td><td>{recipe}</td><td>{100 * v['top1']:.1f}%</td>"
                    f"<td>{100 * v['agreement']:.1f}%</td></tr>"
                )
        parts.append("</table></div>")
    parts.append("</main></body></html>")
    return "".join(parts)


def write(result_paths: list[Path], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load(result_paths)
    pngs = figures(runs[0], out_dir)
    abl_path = Path("results/calibration_ablation_dev.json")
    ablation = json.loads(abl_path.read_text()) if abl_path.exists() else None
    report = out_dir / "report.html"
    report.write_text(html(runs, pngs, ablation), encoding="utf-8")
    return [*pngs, report]
