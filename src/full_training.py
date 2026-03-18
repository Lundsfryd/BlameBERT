import os
import torch
from pathlib import Path
from training_pipeline_with_viz import model_trainer, read_best_weighted_bce
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
    {"data_path": training_data_path, "learning_rate": 1e-4, "model_name": "full_training_lr_1e-4"},
    {"data_path": training_data_path, "learning_rate": 1e-5, "model_name": "full_training_lr_1e-5"},
    {"data_path": training_data_path, "learning_rate": 1e-6, "model_name": "full_training_lr_1e-6"}]



overall_bwbe = float("inf") #artbitrarily high to ensure all from here is smaller
best_model_path = None
best_report_path = None


######## RUN TRAINING LOOOP HERE ###########
for t_args in training_data_args[:]:
    print(f"\n\n#### Training on {t_args['model_name']} ####\n\n")

    report_path = os.path.join(output_dir,F"model_on_{t_args['model_name']}_report.txt")


    model = model_trainer(data_input_path = t_args["data_path"],
        output_dir = output_dir,
        model_name = t_args["model_name"], 
        save_model = True,
        subset = None,
        report_path=report_path,  # new argument
        batch_size=256,
        learning_rate=float(t_args["learning_rate"]),
        )

    current_wbce = read_best_weighted_bce(Path(output_dir), model_name = t_args['model_name'])

    if current_wbce < overall_bwbe:
        overall_bwbe = current_wbce
        best_model_path = os.path.join(output_dir, "full_models", t_args["model_name"])
        best_report_path = report_path

    del model
    torch.cuda.empty_cache()
    gc.collect()


#run validation on best model
print("running test")

detector = BlameDetector(
    model_path=best_model_path,
    max_length=512,
    batch_size=256,
    model_from_path = True
)

detector.run_validation(validation_data_path, best_report_path)
del detector