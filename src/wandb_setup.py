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
    # FIX 2: reset index before assigning sample_id to avoid leaking pandas index
    balanced_df = balanced_df.reset_index(drop=True)
    balanced_df["sample_id"] = range(len(balanced_df))

    dataset = Dataset.from_pandas(balanced_df)
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

    if wandb.run is None:
        wandb.init(project=project_name, name=(run_name or "trajectories") + "_trajectories")

    wandb.log({"embedding_trajectories": table})
    wandb.finish()



'''# import modules
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
                # Extract sample_id BEFORE filtering the batch
                sample_ids.extend(batch["sample_id"].cpu().numpy())

                # Only pass model-compatible keys to the model
                model_inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

                outputs = model(**model_inputs, output_hidden_states=True)
                cls_emb = outputs.hidden_states[-1][:, 0, :]

                embeddings.append(cls_emb.cpu())
                labels.extend(model_inputs["labels"].cpu().numpy())

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


def build_trajectory_table(history, labels):
    all_points = []
    point_meta = []

    for sid, traj in history.items():
        for t, emb in enumerate(traj):
            all_points.append(emb)
            point_meta.append((sid, t, labels[sid]))  # attach label here

    reducer = umap.UMAP(n_components=2, random_state=42)
    projected = reducer.fit_transform(np.array(all_points))

    table = wandb.Table(columns=["x", "y", "sample_id", "step", "label"])

    for (sid, t, label), (x, y) in zip(point_meta, projected):
        table.add_data(x, y, sid, t, label)

    wandb.log({"embedding_trajectories": table}, step=0)'''