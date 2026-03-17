"""
plot_embeddings.py
──────────────────
Plot 2-D embeddings exported from wandb as CSVs, colour-coded by label.

Two main functions
------------------
plot_embeddings_from_csv()
    Plot a single CSV as a standalone figure.

plot_all_layers()
    Given a directory containing subfolders per layer, produce a 4×4 grid:
    rows = layers (embedding, encoder_layer_04, _08, _11)
    cols = snapshots (pre-training, epoch_1, epoch_2, epoch_3)

Usage
-----
    from plot_embeddings import plot_all_layers
    plot_all_layers("wandb_exports/media/table/layer_embeddings_table",
                    output_path="figures/all_layers.png")

    # or from the command line:
    python plot_embeddings.py grid path/to/layer_embeddings_table/ -o figures/all_layers.png
    python plot_embeddings.py single path/to/file.csv -o figures/single.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ── colour palette ────────────────────────────────────────────────────────────

DEFAULT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

DEFAULT_COLOR_MAP = {
    "0": DEFAULT_COLORS[0],
    "1": DEFAULT_COLORS[1],
}

# ── low-level: plot onto an existing axes ─────────────────────────────────────

def _plot_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    x_col: str = "x",
    y_col: str = "y",
    label_col: str = "label",
    color_map: dict | None = None,
    alpha: float = 0.7,
    point_size: int = 15,
) -> None:
    """Render a single scatter plot onto *ax*."""
    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    unique_labels = sorted(df[label_col].unique())

    if color_map is None:
        color_map = {
            lbl: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            for i, lbl in enumerate(unique_labels)
        }

    for lbl in unique_labels:
        mask = df[label_col] == lbl
        ax.scatter(
            df.loc[mask, x_col],
            df.loc[mask, y_col],
            c=color_map.get(lbl, "grey"),
            label=f"Label {lbl}",
            alpha=alpha,
            s=point_size,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("dim 1", fontsize=8)
    ax.set_ylabel("dim 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


# ── single-file public API ────────────────────────────────────────────────────

def plot_embeddings_from_csv(
    csv_path: str | Path,
    x_col: str = "x",
    y_col: str = "y",
    label_col: str = "label",
    epoch_col: str = "epoch",
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    alpha: float = 0.7,
    point_size: int = 20,
    color_map: dict | None = None,
) -> None:
    """
    Load a single wandb-exported CSV and produce a standalone scatter plot
    colour-coded by label.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    for col in [x_col, y_col, label_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

    title = "Embedding space"
    if epoch_col in df.columns:
        epochs = df[epoch_col].unique()
        title = f"Embedding space — {', '.join(str(e) for e in epochs)}"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_on_ax(ax, df, title=title, x_col=x_col, y_col=y_col,
                label_col=label_col, color_map=color_map,
                alpha=alpha, point_size=point_size)
    ax.legend(title="Label", markerscale=1.5, framealpha=0.8)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[plot_embeddings] Saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ── multi-layer grid ──────────────────────────────────────────────────────────

# Maps folder name → human-readable row label, in display order
LAYER_ORDER = [
    ("embedding_layer",  "Embedding layer"),
    ("encoder_layer_04", "Encoder layer 4"),
    ("encoder_layer_08", "Encoder layer 8"),
    ("encoder_layer_11", "Encoder layer 11"),
]

# Maps filename stem → human-readable column label, in display order
SNAPSHOT_ORDER = [
    ("pre-training", "Pre-training"),
    ("epoch_1",      "Epoch 1"),
    ("epoch_2",      "Epoch 2"),
    ("epoch_3",      "Epoch 3"),
]


def plot_all_layers(
    base_dir: str | Path,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (18, 16),
    alpha: float = 0.65,
    point_size: int = 12,
    color_map: dict | None = None,
    x_col: str = "x",
    y_col: str = "y",
    label_col: str = "label",
) -> None:
    """
    Build a 4-row × 4-column grid of embedding scatter plots.

    Layout
    ------
    Rows    : embedding_layer, encoder_layer_04, encoder_layer_08, encoder_layer_11
    Columns : pre-training, epoch_1, epoch_2, epoch_3

    Parameters
    ----------
    base_dir    : directory containing the layer subfolders, e.g.:
                      base_dir/
                        embedding_layer/
                        encoder_layer_04/
                        encoder_layer_08/
                        encoder_layer_11/
    output_path : save path; if None the figure is shown interactively
    """
    base_dir = Path(base_dir)

    if color_map is None:
        color_map = DEFAULT_COLOR_MAP

    n_rows = len(LAYER_ORDER)
    n_cols = len(SNAPSHOT_ORDER)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        constrained_layout=True,
    )

    for row_idx, (folder_name, layer_label) in enumerate(LAYER_ORDER):
        layer_dir = base_dir / folder_name

        for col_idx, (snapshot_stem, snapshot_label) in enumerate(SNAPSHOT_ORDER):
            ax = axes[row_idx][col_idx]

            # Glob loosely — wandb may append extra suffixes to the filename
            matches = list(layer_dir.glob(f"*{snapshot_stem}*.csv"))

            if not matches:
                ax.set_visible(False)
                print(f"[plot_all_layers] Missing: {layer_dir}/{snapshot_stem}*.csv")
                continue

            df = pd.read_csv(matches[0])

            missing_cols = [c for c in [x_col, y_col, label_col] if c not in df.columns]
            if missing_cols:
                ax.set_visible(False)
                print(f"[plot_all_layers] Columns {missing_cols} not found in {matches[0]}")
                continue

            _plot_on_ax(
                ax, df,
                title=f"{layer_label} · {snapshot_label}",
                x_col=x_col, y_col=y_col, label_col=label_col,
                color_map=color_map, alpha=alpha, point_size=point_size,
            )

    # Single shared legend below the grid
    legend_handles = [
        mpatches.Patch(color=color_map.get("0", DEFAULT_COLORS[0]), label="Label 0"),
        mpatches.Patch(color=color_map.get("1", DEFAULT_COLORS[1]), label="Label 1"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=11,
        title="Label",
        title_fontsize=11,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "mmBERT [CLS] embeddings across layers and training stages",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[plot_all_layers] Saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot wandb embedding CSVs")
    subparsers = parser.add_subparsers(dest="command")

    single = subparsers.add_parser("single", help="Plot a single CSV file")
    single.add_argument("csv_path", type=Path)
    single.add_argument("--output", "-o", type=Path, default=None)

    grid = subparsers.add_parser("grid", help="Plot 4×4 layer/epoch grid")
    grid.add_argument(
        "base_dir", type=Path,
        help="Directory containing embedding_layer/, encoder_layer_XX/ subfolders"
    )
    grid.add_argument("--output", "-o", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "single":
        plot_embeddings_from_csv(args.csv_path, output_path=args.output)
    elif args.command == "grid":
        plot_all_layers(args.base_dir, output_path=args.output)
    else:
        parser.print_help()
