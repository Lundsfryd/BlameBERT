import os
from blame_detection import BlameDetector


#setup paths
parent_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(parent_dir)
inference_data = os.path.join(root_dir, "data", "inference","inference_data.jsonl")
output_path = os.path.join(os.path.dirname(inference_data), "predicted_inference_data.jsonl")
model_dir = os.path.join(root_dir, "training_output", "full_models","data_5__lr0.0001__sched-linear__alpha-sqrt")

detector = BlameDetector(
        model_path=model_dir,
        max_length=512,
        batch_size=128,
        model_from_path=True
    )

detector.predict_from_jsonl_to_file(input_path = inference_data, output_path = output_path, text_key='text',
                                   output_key='prediction', confidence_key='confidence',
                                   show_progress=True)