# %%
import os
from pathlib import Path

from training_pipeline_with_viz import model_trainer #OBS change if not working back to training_pipe
from blame_detection import BlameDetector
import torch


base_dir = Path("/work/MarkusLundsfrydJensen#1865/data_outside_git")

validation_data_path = os.path.join(base_dir,
                            "training_data",
                            "validation_set.jsonl")

train_data_dir = os.path.join(base_dir,
                                "training_data",
                                "diff_hypothesis_agreemen")

output_dir = os.path.join(base_dir,
                        "training_output")

datasets = [ # Obviously five different datasets in future, doing this to test the loop
    {"path": Path(os.path.join(train_data_dir, "1_5_agreement.jsonl" )), "model_name": "data_1_5"},
    {"path": Path(os.path.join(train_data_dir, "2_5_agreement.jsonl" )), "model_name": "data_2_5"},
    {"path": Path(os.path.join(train_data_dir, "3_5_agreement.jsonl" )), "model_name": "data_3_5"},
    {"path": Path(os.path.join(train_data_dir, "4_5_agreement.jsonl" )), "model_name": "data_4_5"},
    {"path": Path(os.path.join(train_data_dir, "5_agreement.jsonl" )), "model_name": "data_5"},
]

######## RUN TRAINING LOOOP HERE ###########
for ds in datasets[:1]:
    print(f"\n\n#### Training on {ds['model_name']} ####\n\n")

    report_path = os.path.join(output_dir,F"model_on_{ds["model_name"]}_report.txt")


    model = model_trainer(data_input_path = ds["path"],
        output_dir = output_dir,
        model_name = ds["model_name"], 
        save_model = False,
        subset = 10000,
        report_path=report_path,  # new argument
        batch_size=64,
        learning_rate=1e-5,
        )

    #run validation on best model
    print("running validation")

    detector = BlameDetector(
        model_path=model,
        max_length=512,
        batch_size=64,
        model_from_path = False
    )

    detector.run_validation(validation_data_path, report_path)
