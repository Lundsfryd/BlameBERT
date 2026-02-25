'''
To do:
- Random sampling rather than time based
- DaCy segmentation
- Larger training set?
- No more effective batch sizing, actual batching at 256
- Slightly larger validation set (500?)
'''

# %%
import torch
import torch.nn as nn
import transformers
import numpy as np
import pandas as pd
import keras
import json
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, recall_score, classification_report, confusion_matrix

# %%
class ModelInstantiation():
   def __init__(self, base_model_name=None, tokenizer=None, access_token=None):
      self.base_model_name = base_model_name if base_model_name else "jhu-clsp/mmBERT-base" # For defaulting to BERT
      self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name) if tokenizer is None else tokenizer
      with open('/work/Bachelor/hugging_api.txt', 'r') as f: # Currently inflexible, should be changed to allow for access token to be None as well
         self.access_token = f.read().strip()

      base_model = AutoModelForSequenceClassification.from_pretrained(
        self.base_model_name, # Full precision
        num_labels=2,
        device_map="auto",
        token = self.access_token # Watch me store my private access token in a public repo
        )

      lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear"
        )

      lora_model = prepare_model_for_kbit_training(base_model)
      for name, param in lora_model.named_parameters():
        if 'lora' in name.lower():
           param.requires_grad = True

      self.lora_model = get_peft_model(lora_model, lora_config)
      self.lora_model.print_trainable_parameters()

   def tokenize_function(self, examples):
      encoded = self.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
      )
      return encoded

   def training_setup(self, output_path_checkpoints):
      self.training_args = TrainingArguments(
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
        load_best_model_at_end=True, # Loading best model based on standard f1
        metric_for_best_model="f1",
        greater_is_better=True,
      )

   def weighted_bincrossentropy(self, true, pred, train_data):
      label_counts = train_data['labels'].value_counts()
      total = len(train_data)
      weight_for_0 = total / (2 * label_counts[0])
      weight_for_1 = total / (2 * label_counts[1])

      """
      Calculates weighted binary cross entropy. The weights are fixed to represent class imbalance in the dataset.

      For example if there are 10x as many positive classes as negative classes,
          if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives
          will be penalized 10 times as much as false negatives.
      """
      bin_crossentropy = binary_crossentropy(true, pred)
      weights = true * weight_for_1 + (1. - true) * weight_for_0
      weighted_bin_crossentropy = weights * bin_crossentropy

      return np.mean(weighted_bin_crossentropy)

   def compute_metrics(self, eval_pred):
      predictions, labels = eval_pred
      probs_2d = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
      probs = probs_2d[:, 1]

      weighted_bce = self.weighted_bincrossentropy(labels, probs, train_data=None)  # pass train_data as needed
      keras_bce = binary_crossentropy(labels, probs)
      keras_bce = float(np.mean(keras_bce.numpy()))

      return {
          'keras_BCE': keras_bce,
          'weighted BCE': weighted_bce, 
          'recall': float(recall_score(labels, probs.round())),
          'precision': float(average_precision_score(labels, probs.round())),  # OBS CHANGE THIS
          'accuracy': float(accuracy_score(labels, probs.round())),
          'f1': float(f1_score(labels, probs.round(), average='macro')),
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
def load_data(model_instance, input_path_val, input_path_train):
   val_data = pd.read_json(input_path_val)
   val_data = val_data[['text', 'label']]
   val_data.rename(columns={'label': 'labels'}, inplace=True)
   val_dataset = Dataset.from_pandas(val_data)
   tokenized_val = val_dataset.map(model_instance.tokenize_function, batched=True, num_proc=16)

   train_data = pd.read_json(input_path_train)
   train_data = train_data[['text', 'label']]
   train_data.rename(columns={'label': 'labels'}, inplace=True)
   train_dataset = Dataset.from_pandas(train_data)
   tokenized_train = train_dataset.map(model_instance.tokenize_function, batched=True, num_proc=16)

   return tokenized_val, tokenized_train


# %%
def main():
   model = ModelInstantiation()

   # Loading data, improve filepaths with path objects going forward - for testing i am using the same data twice
   tokenized_val, tokenized_train = load_data(model, "/work/Bachelor/Bachelor_project/Model_data/validation_set.json", "/work/Bachelor/Bachelor_project/Model_data/validation_set.json")

   # Init training arguments and weigthed trainer
   model.training_setup("../../output")

   pass

# %%
if __name__ == "__main__":
  main()