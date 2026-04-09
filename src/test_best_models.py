#import modules
from blame_detection import BlameDetector
import os

#setup paths
parent_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(parent_dir)
data_dir = os.path.join(root_dir, "data","training_data", "diff_hypothesis_agreemen")

test_data_path = os.path.join(os.path.dirname(data_dir),
                                    "validation_set",
                                    "validation_set.jsonl")

output_dir = os.path.join(root_dir, "training_output")
best_full_models = os.path.join(output_dir, "full_models")


best_models = [os.path.join(best_full_models, "data_1_5__lr0.0001__sched-linear__alpha-sqrt"),
                            os.path.join(best_full_models, "data_2_5__lr0.0001__sched-linear__alpha-sqrt"),
                            os.path.join(best_full_models,"data_3_5__lr0.0005__sched-linear__alpha-sqrt"),
                            os.path.join(best_full_models,"data_4_5__lr0.0001__sched-linear__alpha-sqrt"),
                            os.path.join(best_full_models,"data_5__lr0.0001__sched-linear__alpha-sqrt")
                            ]


for model_path in best_models:
    data_part = os.path.basename(model_path).split("__")[0]
    report_path = os.path.join(output_dir, f"test_report_{data_part}.txt")

    detector = BlameDetector(
        model_path=model_path,
        max_length=512,
        batch_size=32,
        model_from_path=True
    )

    detector.run_validation(test_data_path, report_path)