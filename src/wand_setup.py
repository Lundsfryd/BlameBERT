# import modules
import pandas as pd
from datasets import Dataset
import umap
import numpy as np
import wandb
from transformers import TrainerCallback
import torch

def create_balanced_probe(dataset, n_samples=300):
    df = dataset.to_pandas()

    n_per_class = n_samples // 2

    df_0 = df[df["labels"] == 0].sample(n=min(n_per_class, len(df[df["labels"] == 0])), random_state=42)
    df_1 = df[df["labels"] == 1].sample(n=min(n_per_class, len(df[df["labels"] == 1])), random_state=42)

    balanced_df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)
    balanced_df["sample_id"] = range(len(balanced_df))

    dataset = Dataset.from_pandas(balanced_df)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "sample_id"]
    )

    return dataset


def setup_wandb_embedding_tracker_with_trajectories(model, tokenizer, probe_dataset, device):

    history = {}  # stores embeddings per sample_id across epochs

    def extract_embeddings(dataset):
        model.eval()
        embeddings = []
        labels = []
        sample_ids = []

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

                outputs = model(**batch, output_hidden_states=True)
                cls_emb = outputs.hidden_states[-1][:, 0, :]

                embeddings.append(cls_emb.cpu())
                labels.extend(batch["labels"].cpu().numpy())

                sample_ids.extend(batch["sample_id"].cpu().numpy())

        embeddings = torch.cat(embeddings).numpy()
        return embeddings, labels, sample_ids

    def log_embeddings(epoch):
        embeddings, labels, sample_ids = extract_embeddings(probe_dataset)

        table = wandb.Table(
            columns=["embedding", "label", "sample_id", "epoch"]
        )

        for emb, label, sid in zip(embeddings, labels, sample_ids):

            # store trajectory
            if sid not in history:
                history[sid] = []
            history[sid].append(emb)

            table.add_data(emb.tolist(), int(label), int(sid), int(epoch))

        wandb.log({"embedding_space": table}, step=epoch)

    return log_embeddings, history


class EmbeddingCallback(TrainerCallback):
    def __init__(self, log_fn):
        self.log_fn = log_fn

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        self.log_fn(epoch)


def build_trajectory_table(history):

    all_points = []
    point_meta = []

    for sid, traj in history.items():
        for t, emb in enumerate(traj):
            all_points.append(emb)
            point_meta.append((sid, t))

    reducer = umap.UMAP(n_components=2, random_state=42)
    projected = reducer.fit_transform(np.array(all_points))

    table = wandb.Table(columns=["x", "y", "sample_id", "step"])

    for (sid, t), (x, y) in zip(point_meta, projected):
        table.add_data(x, y, sid, t)

    wandb.log({"embedding_trajectories": table})