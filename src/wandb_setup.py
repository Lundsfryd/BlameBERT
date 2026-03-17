import pandas as pd
from datasets import Dataset
import umap
import numpy as np
import wandb
from transformers import TrainerCallback
import torch
import os
#import plotly.graph_objects as go


def create_balanced_probe(dataset, n_samples=300):
    df = dataset.to_pandas()
    n_per_class = n_samples // 2

    df_0 = df[df["labels"] == 0].sample(n=min(n_per_class, len(df[df["labels"] == 0])), random_state=42)
    df_1 = df[df["labels"] == 1].sample(n=min(n_per_class, len(df[df["labels"] == 1])), random_state=42)

    balanced_df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)
    # FIX 2: reset index before assigning sample_id to avoid leaking pandas index
    balanced_df = balanced_df.reset_index(drop=True)
    balanced_df["sample_id"] = range(len(balanced_df))

    dataset = Dataset.from_pandas(balanced_df, preserve_index=False)  # add this flag
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "sample_id"]
    )
    return dataset


def setup_wandb_embedding_tracker_with_trajectories(model, tokenizer, probe_dataset, device):

    history = {}

    def extract_embeddings(dataset):
        model.eval()
        embeddings = []
        labels = []
        sample_ids = []

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        with torch.no_grad():
            for batch in dataloader:
                # FIX 3: cast sample_id to plain Python int immediately
                sample_ids.extend([int(x) for x in batch["sample_id"].cpu().numpy()])

                model_inputs = {k: v.to(device) for k, v in batch.items()
                                if k in ["input_ids", "attention_mask", "labels"]}

                outputs = model(**model_inputs, output_hidden_states=True)

                # FIX 5: hidden_states can be None on some PEFT-wrapped models
                if outputs.hidden_states is not None:
                    hidden = outputs.hidden_states[-1]
                else:
                    hidden = model.base_model(
                        **model_inputs, output_hidden_states=True
                    ).hidden_states[-1]

                cls_emb = hidden[:, 0, :]
                embeddings.append(cls_emb.cpu())
                labels.extend([int(x) for x in model_inputs["labels"].cpu().numpy()])

        embeddings = torch.cat(embeddings).numpy()
        return embeddings, labels, sample_ids

    def log_embeddings(epoch):
        embeddings, labels, sample_ids = extract_embeddings(probe_dataset)

        table = wandb.Table(columns=["embedding", "label", "sample_id", "epoch"])

        for emb, label, sid in zip(embeddings, labels, sample_ids):
            if sid not in history:
                history[sid] = []
            history[sid].append((emb, label))
            table.add_data(emb.tolist(), int(label), int(sid), int(epoch))

        # FIX 1: don't specify step here — let W&B assign it, or use epoch+1
        # to keep steps monotonically increasing and leave room for the trajectory
        # table which will be logged after training at a higher step
        wandb.log({"embedding_space": table})

    return log_embeddings, history


class EmbeddingCallback(TrainerCallback):
    def __init__(self, log_fn):
        self.log_fn = log_fn

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        self.log_fn(epoch)


def build_trajectory_table(history, project_name="mmbert-danish-politics", run_name=None):
    all_points = []
    point_meta = []

    for sid, traj in history.items():
        for t, (emb, label) in enumerate(traj):
            all_points.append(emb)
            point_meta.append((int(sid), t, int(label)))

    if len(all_points) < 20:
        print(f"Skipping trajectory UMAP: only {len(all_points)} points (need ≥20)")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    projected = reducer.fit_transform(np.array(all_points))

    table = wandb.Table(columns=["x", "y", "sample_id", "step", "label"])
    for (sid, t, label), (x, y) in zip(point_meta, projected):
        table.add_data(float(x), float(y), sid, t, label)

    # Filter nulls before export
    df = pd.DataFrame(table.data, columns=table.columns)
    df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)
    df["step"] = df["step"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)
    df.to_csv("embedding_trajectories.csv", index=False)

    if wandb.run is None:
        wandb.init(project=project_name, name=(run_name or "trajectories") + "_trajectories")

    wandb.log({"embedding_trajectories": table})
    wandb.finish()

def build_trajectory_plot_from_csv(csv_path, output_path=None, max_samples=100):
    import pandas as pd
    from plotly.subplots import make_subplots

    df = pd.read_csv(csv_path)
    df = df[df["label"].notna()]
    sample_ids = df["sample_id"].unique()[:max_samples]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Epoch 0 → 1", "Epoch 1 → 2")
    )

    def add_epoch_traces(fig, step_start, step_end, col):
        for sid in sample_ids:
            sample = df[df["sample_id"] == sid].sort_values("step")
            label = sample["label"].iloc[0]
            color = "#3B8BD4" if label == 0 else "#E8593C"

            segment = sample[sample["step"].isin([step_start, step_end])]
            if len(segment) < 2:
                continue

            fig.add_trace(go.Scatter(
                x=segment["x"],
                y=segment["y"],
                mode="lines+markers",
                line=dict(color=color, width=0.8),
                marker=dict(
                    size=[4, 12],  # small = start epoch, large = end epoch
                    color=color
                ),
                showlegend=False,
                hovertemplate=f"sample_id: {sid}<br>label: {label}<br>step: %{{text}}",
                text=segment["step"].tolist()
            ), row=1, col=col)

        # Legend dummy traces only on first subplot to avoid duplicates
        if col == 1:
            for label, color, name in [(0, "#3B8BD4", "label 0"), (1, "#E8593C", "label 1")]:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(color=color, size=8),
                    name=name
                ), row=1, col=col)

    add_epoch_traces(fig, step_start=0, step_end=1, col=1)
    add_epoch_traces(fig, step_start=1, step_end=2, col=2)

    fig.update_layout(
        title="Embedding trajectories per epoch transition",
        plot_bgcolor="white",
        width=1600,
        height=700
    )
    fig.update_xaxes(title_text="UMAP dim 1")
    fig.update_yaxes(title_text="UMAP dim 2")

    if output_path is None:
        output_path = csv_path.replace(".csv", "_plot.html")

    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")
    return fig