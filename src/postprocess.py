"""Post-processing for the rice grain pipeline."""
from __future__ import annotations

import cv2
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

FULL_HEX = "#00c800"
BROKEN_HEX = "#ff0000"
FULL_RGB = (0, 200, 0)
BROKEN_RGB = (255, 0, 0)

def compute_grain_summary(df: pd.DataFrame, ref_length: float | None = None) -> dict:
    """Count grains and compute full/broken ratios from the feature DataFrame."""
    if "Class" not in df.columns:
        raise ValueError("df has no 'Class' column - run classify_grains(df) first")

    total = len(df)
    n_full = int((df["Class"] == "Full").sum())
    n_broken = total - n_full

    full_pct = 100.0 * n_full / total if total else 0.0
    broken_pct = 100.0 * n_broken / total if total else 0.0
    full_to_broken = n_full / n_broken if n_broken else float("inf")

    return {
        "total": total,
        "full": n_full,
        "broken": n_broken,
        "full_pct": full_pct,
        "broken_pct": broken_pct,
        "full_to_broken": full_to_broken,
        "ref_length": ref_length,
    }


def summary_to_table(summary: dict) -> pd.DataFrame:
    """Turn the summary dict into a 2-column table (Metric / Value)."""
    ratio = summary["full_to_broken"]
    ratio_txt = "inf (no broken)" if np.isinf(ratio) else f"{ratio:.2f} : 1"

    rows = [
        ("Total grains", summary["total"]),
        ("Total full grains", summary["full"]),
        ("Total broken grains", summary["broken"]),
        ("Full ratio", f"{summary['full_pct']:.1f} %"),
        ("Broken ratio", f"{summary['broken_pct']:.1f} %"),
        ("Full : Broken", ratio_txt),
    ]
    if summary.get("ref_length"):
        rows.append(("Reference length", f"{summary['ref_length']:.1f} px"))
    return pd.DataFrame(rows, columns=["Metric", "Value"])

def grain_orientation(cnt: np.ndarray) -> float:
    """Angle (degrees, counter-clockwise on screen) of the grain's major axis.

    Computed from image moments, so the ID text can be rotated to lie along
    the grain. Result is in [-90, 90) so the text is never upside down.
    """
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return 0.0
    theta = 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])  
    return float(-np.degrees(theta))


def draw_class_overlay(
    img: np.ndarray,
    contours: list,
    df: pd.DataFrame,
    alpha: float = 0.6,
) -> np.ndarray:
    """Fill every grain green (Full) or red (Broken) on the raw image. Returns RGB."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()

    for _, row in df.iterrows():
        cnt = contours[int(row["ID"])]
        color = FULL_RGB if row["Class"] == "Full" else BROKEN_RGB
        cv2.drawContours(overlay, [cnt], -1, color, thickness=-1)

    output = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)

    for _, row in df.iterrows():
        cnt = contours[int(row["ID"])]
        color = FULL_RGB if row["Class"] == "Full" else BROKEN_RGB
        cv2.drawContours(output, [cnt], -1, color, 1)
    return output


def _plot_labeled_map(ax, img, contours, df, font_size=8.0, crop_margin=20) -> None:
    ax.imshow(draw_class_overlay(img, contours, df))

    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    for _, row in df.iterrows():
        cnt = contours[int(row["ID"])]
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        ax.text(
            cx, cy, str(int(row["ID"])),
            rotation=grain_orientation(cnt),
            rotation_mode="anchor",
            ha="center", va="center",
            fontsize=font_size, fontweight="bold",
            color="white", path_effects=outline,
        )

    if crop_margin is not None and len(df):
        pts = np.vstack([contours[int(i)].reshape(-1, 2) for i in df["ID"]])
        x, y, w, h = cv2.boundingRect(pts)
        H, W = img.shape[:2]
        ax.set_xlim(max(x - crop_margin, 0), min(x + w + crop_margin, W))
        ax.set_ylim(min(y + h + crop_margin, H), max(y - crop_margin, 0))

    ax.legend(
        handles=[Patch(facecolor=FULL_HEX, label="Full"),
                 Patch(facecolor=BROKEN_HEX, label="Broken")],
        loc="upper right", fontsize=10, framealpha=0.9,
    )
    ax.set_title(f"Separated labeled grain map ({len(df)} grains)", fontsize=13)
    ax.axis("off")

def _plot_length_width_scatter(ax, df, ref_length=None, broken_ratio=0.75) -> None:
    for cls, color in (("Full", FULL_HEX), ("Broken", BROKEN_HEX)):
        sub = df[df["Class"] == cls]
        ax.scatter(sub["Length"], sub["Width"], s=22, c=color,
                   alpha=0.75, edgecolors="k", linewidths=0.3,
                   label=f"{cls} ({len(sub)})")

    if ref_length:
        threshold = broken_ratio * ref_length
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1,
                   label=f"Broken threshold = {threshold:.1f}px")

    ax.set_xlabel("Length (px)", fontsize=9)
    ax.set_ylabel("Width (px)", fontsize=9)
    ax.set_title("Length vs Width", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def _plot_broken_summary(ax, summary) -> None:
    labels = ["Full", "Broken"]
    counts = [summary["full"], summary["broken"]]
    pcts = [summary["full_pct"], summary["broken_pct"]]
    bars = ax.bar(labels, pcts, color=[FULL_HEX, BROKEN_HEX],
                  edgecolor="k", linewidth=0.5)

    for bar, n, p in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{p:.1f}% (n={n})", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 115)
    ax.set_ylabel("Share of grains (%)", fontsize=9)
    ax.set_title(f"Broken grain percentage: {summary['broken_pct']:.1f}%", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def _plot_table(ax, table_df: pd.DataFrame) -> None:
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (row, _), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dddddd")
    ax.set_title("Summary table", fontsize=11)

def show_postprocess(
    img: np.ndarray,
    contours: list,
    df: pd.DataFrame,
    ref_length: float | None = None,
    broken_ratio: float = 0.75,
    save_path: str | None = None,
    label_font_size: float = 8.0,
    crop_margin: int | None = 20,
) -> pd.DataFrame:
  
    """Build the post-processing figure + summary table."""

    if len(df) == 0:
        print("No grains detected - nothing to post-process.")
        return pd.DataFrame(columns=["Metric", "Value"])

    summary = compute_grain_summary(df, ref_length)
    table_df = summary_to_table(summary)

    print("\n===== POST-PROCESS SUMMARY =====")
    print(table_df.to_string(index=False))


    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 0.8],
                  height_ratios=[1, 0.9, 1])
    ax_map = fig.add_subplot(gs[:, :2])
    ax_scatter = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 2])
    ax_table = fig.add_subplot(gs[2, 2])

    _plot_labeled_map(ax_map, img, contours, df, label_font_size, crop_margin)
    _plot_length_width_scatter(ax_scatter, df, ref_length, broken_ratio)
    _plot_broken_summary(ax_bar, summary)
    _plot_table(ax_table, table_df)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return table_df
