import os
import torch
from pathlib import Path
from training_pipeline_with_viz import model_trainer
from blame_detection import BlameDetector
import gc

# ------------------------------------------------------------------- # 

# %%
parent_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(parent_dir)
data_dir = os.path.join(root_dir, "data","training_data", "diff_hypothesis_agreemen")

# %%
validation_data_path = os.path.join(os.path.dirname(data_dir),
                            "validation_set",
                            "validation_set.jsonl")
# %%

output_dir = os.path.join(root_dir,
                        "training_output")

training_data_path = Path(os.path.join(data_dir, "4_5_agreement.jsonl" ))

training_data_args = [
    {"data_path": training_data_path, "learning_rate": 1e-4 "model_name": "full_training_lr_1e-4"},
    {"data_path": training_data_path, "learning_rate": 1e-5 "model_name": "full_training_lr_1e-5"},
    {"data_path": training_data_path, "learning_rate": 1e-6 "model_name": "full_training_lr_1e-6"}]






######## RUN TRAINING LOOOP HERE ###########
for t_args in training_data_args[:]:
    print(f"\n\n#### Training on {t_args['model_name']} ####\n\n")

    report_path = os.path.join(output_dir,F"model_on_{t_args["model_name"]}_report.txt")


    model = model_trainer(data_input_path = t_args["data_path"],
        output_dir = output_dir,
        model_name = t_args["model_name"], 
        save_model = True,
        subset = None
        report_path=report_path,  # new argument
        batch_size=256,
        learning_rate=float(t_args["learning_rate"]),
        )

    wbce = 
    del model
    del detector
    torch.cuda.empty_cache()
    gc.collect()


#run validation on best model
    print("running validation")

    detector = BlameDetector(
        model_path=model,
        max_length=512,
        batch_size=256,
        model_from_path = True
    )

    detector.run_validation(validation_data_path, report_path)