'''
To do:
- Random sampling rather than time based
- DaCy segmentation
- Larger training set?
- No more effective batch sizing, actual batching at 256
- Slightly larger validation set (500?)
- 

'''

# %% 
import torch
import transformers
import numpy as np
import pandas as pd
import keras
import json
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, recall_score, classification_report, confusion_matrix

# %% 
# Trying classes
class ModelInstantiation:
   def __init__(self, base_model_name: str, tokenizer: str):
      self.base_model_name = "jhu-clsp/mmBERT-base"
      self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

      base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        dtype=torch.bfloat16, #b addded
        device_map="auto"
        )

      lora_config = LoraConfig(
        r=16,  # Low-rank dimension - we can look into increasing rank to train more info to the model
        lora_alpha=32, # Again alpha scaling is contentious, but some recommend a simple doubling of the rank
        lora_dropout=0.05,
        target_modules="all-linear"  # Fine-tuning all linear (classification, attention... layers)
        )
      
      # Enable gradient computation for LoRA parameters
      lora_model = prepare_model_for_kbit_training(base_model)
      for name, param in lora_model.named_parameters():
        if 'lora' in name.lower():
           param.requires_grad = True   
  
      lora_model = get_peft_model(lora_model, lora_config)
      lora_model.print_trainable_parameters()
    
    def tokenize_function(examples):
      return tokenizer(examples["text"], 
      padding="max_length", 
      truncation=True,
      max_length=512, # Padding to 512 to massively cut down on computation compared to base 8,192 tokens. 
      )







def main():
   pass


 # %%
 def tokenize_function(examples):
    return tokenizer(examples["text"], 
    padding="max_length", 
    truncation=True,
    max_length=512, # Padding to 512 to massively cut down on computation compared to base 8,192 tokens. 
    )