# %%
import torch
import torch.nn as nn
import transformers
import numpy as np
import pandas as pd
import keras
import json
import os
import argparse
from pathlib import Path
from collections import Counter
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import tensorflow as tf
import wandb
from embedding_viz import LayerEmbeddingVizCallback, log_layer_embeddings, create_balanced_probe 

# %%

def main():
    parser = argparse.ArgumentParser(
        prog="mmBERT training pipeline",
        description="LoRA style fine tuning on Danish political texts"
    )

    parser.add_argument("--data_input_path",
                        "-d_in",
                        type=Path,
                        required=True,
                        help="input path to a jsonl file for processing to training dataset")

    parser.add_argument("--validation_input_path",
                        "-v_in",
                        type=Path,
                        required=False,
                        help="input path to a jsonl file used for validating the model")

    parser.add_argument("--output_path",
                        "-o",
                        type=Path,
                        required=True,
                        help="output path to both model checkpoints and full model saving, depending on the save model flag")

    parser.add_argument("--model_name",
                        "-name",
                        type=str,
                        required=True,
                        help="name of the saved model, both full model, checkpoints, and wandb will save with this name")

    parser.add_argument("--save_model",
                        "-s",
                        action="store_true",
                        help="boolean flag to save final model or not, defaults to false")

    parser.add_argument("--learning_rate",
                        "-lr",
                    default=1e-5,
                    required=False)

    parser.add_argument("--subset",
                        "-ss",
                        type=int,
                        default=None,
                        help="Takes an integer which determines the absolute size of the subset dataset. Defaults to None, which means training happens on full dataset.",
                        required=False)

    parser.add_argument("--batch_size",
                        "-bs",
                        type=int,
                        default=64,
                        required=False)

    args = parser.parse_args()

    model_trainer(
        data_input_path=args.data_input_path,
        output_dir=args.output_path,
        model_name=args.model_name,
        save_model=args.save_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset=args.subset
    )


def model_trainer(data_input_path, output_dir, model_name, save_model=False, subset = None, report_path = None, learning_rate = 1e-5, batch_size = 64):

    # -----------------------------------------------------------------------------------------

    model = ModelInstantiation(learning_rate = learning_rate, batch_size = batch_size)
    wandb.init(project="mmbert-danish-politics", name=model_name)

    # FIX 1: Use the output_path directly — do not prepend ../../ which mangles user-provided paths
    output_model_dir = Path(output_dir)
    os.makedirs(output_model_dir, exist_ok=True)

    # Load data function also returns class weights
    print("tokenizing dataset...")
    tokenized_eval, tokenized_train, class_weights = load_data(model, input_path=data_input_path, subset=subset)  
    

    probe_dataset = create_balanced_probe(tokenized_eval, n_samples=300)

    # ── Log the pre-training baseline BEFORE trainer.train() ─────────────────
    log_layer_embeddings(                                                   
        model         = model.lora_model,
        probe_dataset = probe_dataset,
        device        = model.device,
        epoch_label   = "pre-training",
        layers_to_log = [0, 4, 8, 11],  # embed layer + 3 encoder checkpoints
    )

    print("initializing trainer")
    trainer = WeightedLossTrainer(  # Custom trainer function taking class balance into account
        model=model.lora_model,  # LoRA parameters
        args=model.training_setup(
            output_path_checkpoints=str(output_model_dir / "checkpoints" / model_name)
        ),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=model.compute_metrics,  # Custom metrics function
        class_weights=class_weights,  # Weighted by presence in dataset
        data_collator=model.data_collator,
        callbacks=[
            LayerEmbeddingVizCallback(                
                model         = model.lora_model,
                probe_dataset = probe_dataset,
                device        = model.device,
                layers_to_log = [0, 4, 8, 11],       # same layers as baseline
                reduction     = "umap",               # or "pca" for speed
            ),
        ],
    )

    print("trainer.train")
    trainer.train()
    print("training done")

    if report_path is not None:
        
        # Force Keras/TF to CPU to avoid CUDA handle conflict with PyTorch
        with tf.device('/CPU:0'):
            predictions_output = trainer.predict(tokenized_eval)
        
        preds = np.exp(predictions_output.predictions)
        preds = (preds / preds.sum(axis=1, keepdims=True))[:, 1].round().astype(int)
        labels = predictions_output.label_ids

        os.makedirs(Path(report_path).parent, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"=== Best Epoch Report: {model_name} ===\n\n")
            f.write(classification_report(labels, preds))


     #saving
    if save_model:
        save_dir = output_model_dir / "full_models" / model_name
        print(f"\n\n#### Saving full model to {save_dir} ####\n\n")
        os.makedirs(save_dir, exist_ok=True)
        trainer.model.save_pretrained(str(save_dir))
        model.tokenizer.save_pretrained(str(save_dir))
    else:
        # Remove checkpoints directory if we don't want anything saved
        import shutil
        checkpoint_dir = output_model_dir / "checkpoints" / model_name
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
    
    wandb.finish(exit_code = 0)


    return trainer.model


# %%
class ModelInstantiation():
    def __init__(self, learning_rate = 1e-5, batch_size = 64,base_model_name=None, tokenizer=None):
        self.base_model_name = base_model_name if base_model_name else "jhu-clsp/mmBERT-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name) if tokenizer is None else tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        print(f'\n\n#### Running training on {self.device} ####\n\n')

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=2,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules="all-linear"
        )

        self.lora_model = get_peft_model(base_model, lora_config)
        self.lora_model.print_trainable_parameters()
        return

    def tokenize_function(self, examples):
        encoded = self.tokenizer(
            examples["text"],
            #padding="max_length", # Removing for smart batching
            truncation=True,
            max_length=512,
        )
        return encoded

    def training_setup(self, output_path_checkpoints=None): 
      # names the run after model name from argparse
      self.training_args = TrainingArguments(
         report_to="wandb",
         gradient_checkpointing=True,
         output_dir=output_path_checkpoints,
         learning_rate=self.learning_rate,
         num_train_epochs=5,
         per_device_train_batch_size=self.batch_size,
         logging_steps=50,
         #eval_strategy="epoch",
         #save_strategy="epoch",
         eval_strategy="steps",
         save_strategy="steps",
         eval_steps=200,
         save_steps=200,
         save_total_limit=3,
         dataloader_pin_memory=False,
         dataloader_num_workers=8,
         remove_unused_columns=True,
         max_grad_norm=1.0,
         disable_tqdm=False,
         load_best_model_at_end=True,
         metric_for_best_model="weighted_BCE",
         greater_is_better=False,
      )
      return self.training_args

    def _weighted_bincrossentropy_from_labels(self, true, pred):
        """
        Compute weighted BCE using class weights derived directly from the
        provided labels array — no dependency on external training data.
        """
        true = np.asarray(true, dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)

        label_counts = Counter(true.astype(int).tolist())
        total = len(true)

        # FIX 5: Guard against missing classes so we never divide by zero
        weight_for_0 = total / (2 * label_counts[0]) if label_counts[0] > 0 else 1.0
        weight_for_1 = total / (2 * label_counts[1]) if label_counts[1] > 0 else 1.0

        with tf.device('/CPU:0'):                          
            bin_crossentropy = binary_crossentropy(true, pred)
            weights = true * weight_for_1 + (1.0 - true) * weight_for_0
            weighted_bin_crossentropy = weights * bin_crossentropy

        return float(np.mean(weighted_bin_crossentropy))

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        probs_2d = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
        probs = probs_2d[:, 1]  # Keeping only positive class

        # FIX 5 (continued): compute weighted BCE from the eval batch's own label
        # distribution instead of passing train_data=None (which crashed previously)
        weighted_bce = self._weighted_bincrossentropy_from_labels(labels, probs)

        with tf.device('/CPU:0'):                          
            keras_bce = binary_crossentropy(labels.astype(np.float32), probs.astype(np.float32))
            keras_bce = float(np.mean(keras_bce.numpy()))

        return {
            'keras_BCE': keras_bce,
            'weighted_BCE': weighted_bce,
            'recall': float(recall_score(labels, probs.round())),
            'precision': float(precision_score(labels, probs.round())),
            'accuracy': float(accuracy_score(labels, probs.round())),
            'f1': float(f1_score(labels, probs.round(), average='macro')),
            'number_of_true_preds': int(sum(probs.round())),
            'number_of_true_labels': int(sum(labels))
        }


class WeightedLossTrainer(Trainer): #OBS CLASS WEIGHTS ARE NONE?
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# %%
def read_jsonl(file_path):
    """Read a jsonl file and return a list of records."""
    print(f'\n\n#### Reading "{file_path}" for training / test data (90/10 split) ####\n\n')
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_data(model_instance, input_path, subset = None):
    # Loading data, renaming columns to what trainer expects, and converting to Dataset
    data_records = read_jsonl(input_path)
    data = pd.DataFrame(data_records)
    data = data[["text", "label"]]
    data.rename(columns={'label': 'labels'}, inplace=True)

    if subset is not None:
        df_0 = data[data['labels'] == 0].sample(n=subset // 2, random_state=42)
        df_1 = data[data['labels'] == 1].sample(n=subset // 2, random_state=42)
        data = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

    dataset = Dataset.from_pandas(data)

    # 90/10 train/eval split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = dataset["test"]
    train_dataset = dataset["train"]

    # Tokenizing
    tokenized_eval = eval_dataset.map(model_instance.tokenize_function, batched=True)
    tokenized_train = train_dataset.map(model_instance.tokenize_function, batched=True)

    # Sorting eval for smart batching (tokenizer pads to max length, make batches as small as possible)
    sorted_indices_eval = sorted(
    range(len(tokenized_eval)), key=lambda i: len(tokenized_eval[i]["text"].split()), reverse=True)
    tokenized_eval = tokenized_eval.select(sorted_indices_eval)

    # Sorting train for smart batching
    sorted_indices_train = sorted(
    range(len(tokenized_train)), key=lambda i: len(tokenized_train[i]["text"].split()), reverse=True)
    tokenized_train = tokenized_train.select(sorted_indices_train)

    # Computing class weights from training split only
    label_counts = Counter(train_dataset['labels'])
    total = len(train_dataset)
    weight_for_0 = total / (2 * label_counts[0])
    weight_for_1 = total / (2 * label_counts[1])

    return tokenized_eval, tokenized_train, torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)


def read_best_weighted_bce(run_output_dir: Path, model_name: str) -> float:
    """
    Parse the best eval weighted_BCE from the HuggingFace trainer_state.json
    that is written inside the best checkpoint folder.
 
    Falls back to a large sentinel value (1e9) if the file cannot be found,
    so the sweep still completes and the run is ranked last.
    """
    checkpoint_root = run_output_dir / "checkpoints" / model_name
 
    if not checkpoint_root.exists():
        print(f"  [WARNING] checkpoint dir not found: {checkpoint_root}")
        return float("1e9")
 
    # trainer_state.json is in every checkpoint sub-folder; pick the last one.
    state_files = sorted(checkpoint_root.glob("checkpoint-*/trainer_state.json"))
    if not state_files:
        # HF also writes a trainer_state.json directly in the output_dir
        direct = checkpoint_root / "trainer_state.json"
        if direct.exists():
            state_files = [direct]
 
    if not state_files:
        print(f"  [WARNING] no trainer_state.json found under {checkpoint_root}")
        return float("1e9")
 
    # Use the last checkpoint's state (most recent epoch)
    with open(state_files[-1]) as f:
        state = json.load(f)
 
    # Extract the best_metric value recorded by the Trainer
    # (metric_for_best_model = "weighted_BCE", greater_is_better = False)
    best_metric = state.get("best_metric")
    if best_metric is not None:
        return float(best_metric)
 
    # Fallback: scan the log history for the minimum eval/weighted_BCE
    log_history = state.get("log_history", [])
    bce_values = [
        entry["eval_weighted_BCE"]
        for entry in log_history
        if "eval_weighted_BCE" in entry
    ]
    if bce_values:
        return min(bce_values)
 
    print(f"  [WARNING] weighted_BCE not found in trainer_state for {model_name}")
    return float("1e9")

# %%
if __name__ == "__main__":
    main()