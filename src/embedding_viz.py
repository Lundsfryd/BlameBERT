"""
embedding_viz.py
────────────────
Layer-wise hidden-state visualisation for mmBERT LoRA fine-tuning.

What it does
------------
* Captures [CLS] embeddings from *every* encoder layer of mmBERT at three
  stages: before training starts, and at the end of every epoch.
* Reduces them to 2-D with UMAP (falls back to PCA if umap-learn is not
  installed).
* Logs one wandb panel per layer per snapshot, so you can scrub through
  training and watch class separation develop — especially in the upper
  layers where fine-tuning has the most effect.

Usage inside the existing pipeline
------------------------------------
In model_trainer() in your main script, add these lines:

    from embedding_viz import LayerEmbeddingVizCallback, log_layer_embeddings

    # 1. Build probe dataset (already done via create_balanced_probe)
    probe_dataset = create_balanced_probe(tokenized_eval, n_samples=300)

    # 2. Snapshot BEFORE training (epoch-0 baseline)
    log_layer_embeddings(
        model         = model.lora_model,
        tokenizer     = model.tokenizer,
        probe_dataset = probe_dataset,
        device        = model.device,
        epoch_label   = "pre-training",
        layers_to_log = [0, 4, 8, 11],   # None logs all 13 layers
    )

    # 3. Add the callback alongside the existing EmbeddingCallback
    trainer = WeightedLossTrainer(
        ...
        callbacks=[
            EmbeddingCallback(embedding_logger),          # existing
            LayerEmbeddingVizCallback(                    # new
                model         = model.lora_model,
                tokenizer     = model.tokenizer,
                probe_dataset = probe_dataset,
                device        = model.device,
                layers_to_log = [0, 4, 8, 11],
            ),
        ],
    )

Recommended layers_to_log
--------------------------
    [0, 4, 8, 11]   -- embedding layer + early/middle/final encoder layers
    None            -- all 13 layers (slower, but complete picture)

The literature shows fine-tuning affects upper layers most strongly, so layers
8-11 are the most interesting to watch evolve across epochs.

Dependencies
------------
    pip install umap-learn wandb torch transformers scikit-learn
    (umap-learn is optional; PCA from sklearn is used as a silent fallback)
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch
import wandb
from datasets import Dataset
from transformers import PreTrainedModel
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


# ── dimensionality reduction ──────────────────────────────────────────────────

def _reduce_2d(
    embeddings: np.ndarray,
    method: Literal["umap", "pca"] = "umap",
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce an (N, D) embedding matrix to (N, 2) for scatter plotting.
    Tries UMAP first; silently falls back to PCA if umap-learn is absent.
    """
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=random_state,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            warnings.warn(
                "umap-learn is not installed — falling back to PCA for 2-D reduction. "
                "Install with:  pip install umap-learn",
                stacklevel=2,
            )

    # PCA fallback (always available via sklearn)
    from sklearn.decomposition import PCA  # type: ignore

    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(embeddings)


# ── core embedding extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_layer_cls_embeddings(
    model: PreTrainedModel,
    probe_dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    base = model.base_model if hasattr(model, "base_model") else model

    labels = np.array(probe_dataset["labels"], dtype=int)
    n = len(labels)
    n_layers: int = base.config.num_hidden_layers + 1
    accumulated: list[list[np.ndarray]] = [[] for _ in range(n_layers)]

    for start in range(0, n, batch_size):
        input_ids = probe_dataset["input_ids"][start : start + batch_size].to(device)
        attention_mask = probe_dataset["attention_mask"][start : start + batch_size].to(device)

        outputs = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        for layer_idx, layer_hs in enumerate(outputs.hidden_states):
            cls_vec = layer_hs[:, 0, :].cpu().float().numpy()
            accumulated[layer_idx].append(cls_vec)

    all_hidden = np.stack(
        [np.concatenate(batches, axis=0) for batches in accumulated], axis=0
    )
    return all_hidden, labels
# ── wandb logging ─────────────────────────────────────────────────────────────

def log_layer_embeddings(
    model: PreTrainedModel,
    probe_dataset: Dataset,
    device: torch.device,
    epoch_label: str,
    reduction: Literal["umap", "pca"] = "umap",
    layers_to_log: list[int] | None = None,
    batch_size: int = 32,
) -> None:
    """
    Extract [CLS] embeddings from all (or selected) layers and log 2-D scatter
    plots to the currently active wandb run.

    Parameters
    ----------
    epoch_label   : str
        Tag for this snapshot, e.g. "pre-training", "epoch_1", "epoch_3".
    layers_to_log : list[int] | None
        Layer indices to log.  0 = token embedding layer; 1–12 = encoder blocks.
        Pass None to log every layer.
    """
    print(f"\n[embedding_viz] Extracting hidden states — {epoch_label} …")

    all_hidden, labels = extract_layer_cls_embeddings(
        model, probe_dataset, device, batch_size=batch_size
    )
    n_layers = all_hidden.shape[0]

    if layers_to_log is None:
        layers_to_log = list(range(n_layers))

    log_dict: dict = {}

    for layer_idx in layers_to_log:
        if layer_idx >= n_layers:
            warnings.warn(
                f"layers_to_log contains index {layer_idx} but model only has "
                f"{n_layers} layers (0–{n_layers - 1}). Skipping.",
                stacklevel=2,
            )
            continue

        layer_name = (
            "embedding_layer" if layer_idx == 0 else f"encoder_layer_{layer_idx:02d}"
        )

        # 2-D reduction
        embeddings_2d = _reduce_2d(all_hidden[layer_idx], method=reduction)

        # Build a wandb Table for the scatter plot
        table = wandb.Table(columns=["x", "y", "label", "epoch"])
        for (x, y), lbl in zip(embeddings_2d, labels):
            table.add_data(float(x), float(y), int(lbl), epoch_label)

        scatter_key = f"layer_embeddings/{layer_name}/{epoch_label}"
        log_dict[scatter_key] = wandb.plot.scatter(
            table,
            x="x",
            y="y",
            title=f"{layer_name} · {epoch_label}",
        )

        # Also store the raw table so it can be queried / downloaded later
        log_dict[f"layer_embeddings_table/{layer_name}/{epoch_label}"] = table

    wandb.log(log_dict)
    print(
        f"[embedding_viz] Logged {len(layers_to_log)} layer(s) "
        f"for snapshot '{epoch_label}'."
    )


# ── Trainer callback ──────────────────────────────────────────────────────────

class LayerEmbeddingVizCallback(TrainerCallback):
    """
    Hugging Face TrainerCallback that automatically logs layer-wise [CLS]
    embedding scatter plots to wandb at the end of every training epoch.

    The pre-training baseline must be logged separately (before trainer.train())
    via a direct call to log_layer_embeddings(..., epoch_label="pre-training").

    Parameters
    ----------
    model         : the PEFT-wrapped mmBERT model
    tokenizer     : corresponding tokenizer
    probe_dataset : small balanced Dataset (≈300 samples) with 'text' + 'labels'
    device        : torch.device
    reduction     : "umap" (default) or "pca"
    layers_to_log : list of layer indices to visualise, or None for all layers.
                    Recommended: [0, 4, 8, 11] for a fast but informative view.
    batch_size    : batch size for inference during extraction
    """

    def __init__(
        self,
        model: PreTrainedModel,
        probe_dataset: Dataset,
        device: torch.device,
        reduction: Literal["umap", "pca"] = "umap",
        layers_to_log: list[int] | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.probe_dataset = probe_dataset
        self.device = device
        self.reduction = reduction
        self.layers_to_log = layers_to_log
        self.batch_size = batch_size

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        epoch = int(round(state.epoch))
        log_layer_embeddings(
            model=self.model,
            probe_dataset=self.probe_dataset,
            device=self.device,
            epoch_label=f"epoch_{epoch}",
            reduction=self.reduction,
            layers_to_log=self.layers_to_log,
            batch_size=self.batch_size,
        )


# ── optional: re-log from a pre-collected history dict ───────────────────────

def log_trajectory_from_history(
    history: dict[str, np.ndarray],
    labels: np.ndarray,
    reduction: Literal["umap", "pca"] = "umap",
) -> None:
    """
    Re-reduce and log embeddings that were already collected into a history
    dict (e.g. from setup_wandb_embedding_tracker_with_trajectories).

    Expected format
    ---------------
    history = {
        "pre-training": np.ndarray of shape (N, H),
        "epoch_1":      np.ndarray of shape (N, H),
        ...
    }
    Each array should be the [CLS] vector from the model's final layer.
    """
    for snapshot_label, emb in history.items():
        reduced = _reduce_2d(emb, method=reduction)
        table = wandb.Table(columns=["x", "y", "label", "epoch"])
        for (x, y), lbl in zip(reduced, labels):
            table.add_data(float(x), float(y), int(lbl), snapshot_label)
        wandb.log(
            {
                f"trajectory/{snapshot_label}": wandb.plot.scatter(
                    table,
                    x="x",
                    y="y",
                    title=f"[CLS] trajectory — {snapshot_label}",
                )
            }
        )
