# %%
import os
from pathlib import Path

from scripting_training import model_trainer
from blame_detection import BlameDetector

training_data_path = Path("C:/Users/marku/Desktop/test_inference_data.jsonl")
validation_data_path = training_data_path

#make subset

print("starting model trainer")
model = model_trainer(data_input_path = training_data_path,
        output_path = "Data_outside_git", 
        model_name = "test_model", 
        save_model = False)

#run validation on best model
print("running validation")

detector = BlameDetector(
    model_path=model,
    max_length=512,
    batch_size=64,
    model_from_path = False
)

detector.run_validation(validation_data_path)


# %%
