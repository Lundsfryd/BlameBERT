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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# %%
def main():

   parser = argparse.ArgumentParser(
      prog="mmBERT training pipeline",
      description="LoRA style fine tuning on Danish political texts"
   )

   parser.add_argument("--data_input_path",
                     type=Path,
                     required=True)

   parser.add_argument("--validation_input_path",
                     type=Path,
                     required=False) # Should be required when doing actual validation

   parser.add_argument("--output_path",
                     type=Path,
                     required=True)

   parser.add_argument("--model_name",
                     type=str,
                     required=True)

   parser.add_argument("--save_model",
                     action="store_true",
                     help="boolean flag to save final model or not, defaults to false")

   args = parser.parse_args()

# -----------------------------------------------------------------------------------------

   model = ModelInstantiation()

   input_path = args.data_input_path
   output_path = args.output_path
   model_name = args.model_name
   
   # Defining output outside git from argparse input
   output_model_dir = os.path.join("..","..",output_path)
   os.makedirs(output_model_dir, exist_ok=True)
     
   # Load data function also returns class weights
   tokenized_eval, tokenized_train, class_weights = load_data(model, input_path=input_path)

   trainer = WeightedLossTrainer( # Custom trainer function taking class balance into account
      model=model.lora_model, # LoRA parameters
      args=model.training_setup(output_path_checkpoints=f'{output_model_dir}/checkpoints/{model_name}'), # This saves checkpoints to an output folder outside GIT
      train_dataset=tokenized_train,
      eval_dataset=tokenized_eval,
      compute_metrics=model.compute_metrics, # Custom metrics function
      class_weights=class_weights # Weighted by presence in dataset 
   )

   trainer.train()

   # Saving
   if args.save_model:
      print(f"\n\n#### Saving full model to {output_model_dir}/full_models/{model_name} ####\n\n")
      saved_model = trainer.model
      saved_model.save_pretrained(f'{output_model_dir}/full_models/{model_name}')
   else:
      print("\n\n#### No full model saved#### \n\n")


# %%
class ModelInstantiation():
   def __init__(self, base_model_name=None, tokenizer=None, access_token=None):
      self.base_model_name = base_model_name if base_model_name else "jhu-clsp/mmBERT-base" # For defaulting to mmBERT but allowing for other models
      self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name) if tokenizer is None else tokenizer # Using standard mmBERT tokenizer
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Flexible GPU Availability
      print(f'\n\n#### Running training on {self.device} ####\n\n')

      base_model = AutoModelForSequenceClassification.from_pretrained(
        self.base_model_name, # Full precision
        num_labels=2, # Binary classification
        device_map="auto" # Mapping to GPU if available
        )

      lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules="all-linear"
        )

      lora_model = prepare_model_for_kbit_training(base_model) # Preparing for lora
      for name, param in lora_model.named_parameters():
        if 'lora' in name.lower():
           param.requires_grad = True

      self.lora_model = get_peft_model(lora_model, lora_config)
      self.lora_model.print_trainable_parameters()
      return

   def tokenize_function(self, examples):
      encoded = self.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512, # Tokenizing to 512 and truncating above. It looks at 1 sentence at a time.
      )
      return encoded

   def training_setup(self, output_path_checkpoints=None):
      self.training_args = TrainingArguments(
         report_to="wandb",
        output_dir=output_path_checkpoints,
        learning_rate=1e-5,
        num_train_epochs=3,
        per_device_train_batch_size=256, # Actual batching at 256
        logging_steps=1,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        remove_unused_columns=True,
        max_grad_norm=1.0,
        disable_tqdm=False,
        load_best_model_at_end=True, # Loading best model based on weighted BCE on evaluation set (90/10 split)
        metric_for_best_model="weighted_BCE",
        greater_is_better=False,
      )
      return self.training_args

   def weighted_bincrossentropy(self, true, pred, train_data):
      # Calculates weighted binary cross entropy. The weights represent actual imbalance in data.
      label_counts = Counter(train_dataset['labels'])
      total = len(train_dataset)

      # Counting labels and computing weights
      weight_for_0 = total / (2 * label_counts[0])
      weight_for_1 = total / (2 * label_counts[1])
      bin_crossentropy = binary_crossentropy(true, pred)
      weights = true * weight_for_1 + (1. - true) * weight_for_0
      weighted_bin_crossentropy = weights * bin_crossentropy

      return np.mean(weighted_bin_crossentropy)

   def compute_metrics(self, eval_pred):
      predictions, labels = eval_pred
      probs_2d = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
      probs = probs_2d[:, 1] # Keeping only positive class

      weighted_bce = self.weighted_bincrossentropy(labels, probs, train_data=None)
      keras_bce = binary_crossentropy(labels, probs)
      keras_bce = float(np.mean(keras_bce.numpy())) # From eagertensor (keras) to float

      return {
          'keras_BCE': keras_bce,
          'weighted_BCE': weighted_bce, 
          'recall': float(recall_score(labels, probs.round())),
          'precision': float(precision_score(labels, probs.round())),
          'accuracy': float(accuracy_score(labels, probs.round())), # Rounding required as this only takes integers
          'f1': float(f1_score(labels, probs.round(), average='macro')), # Macro f1 is best for imbalanced data
          'number_of_true_preds': sum(probs.round()),
          'number_of_true_labels': sum(labels)
      }

class WeightedLossTrainer(Trainer):
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

def load_data(model_instance, input_path):
   # Loading data, renaming columns to what trainer expects, and converting to Dataset
   data_records = read_jsonl(input_path)
   data = pd.DataFrame(data_records)
   data = data[["text","label"]] # Necessary to convert to dataframe for the following operations
   data.rename(columns={'label': 'labels'}, inplace=True)
   dataset = Dataset.from_pandas(data)

   # Doing test train split
   dataset = dataset.train_test_split(test_size=0.1, seed=42)
   eval_dataset = dataset["test"]
   train_dataset = dataset["train"]

   # Tokenizing
   tokenized_eval = eval_dataset.map(model_instance.tokenize_function, batched=True)
   tokenized_train = train_dataset.map(model_instance.tokenize_function, batched=True)

   # Computing weights
   label_counts = Counter(train_dataset['labels'])
   total = len(train_dataset)
   weight_for_0 = total / (2 * label_counts[0])
   weight_for_1 = total / (2 * label_counts[1]) # Crashes with division by 0 error if no 1 labels, but will not happen in practice

   return tokenized_eval, tokenized_train, torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32) #Class weights


# %%
if __name__ == "__main__":
  main()