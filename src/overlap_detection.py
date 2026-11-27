from blame_detection import BlameDetector

import json
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).parent.parent

#test value
N_ROWS = None

predictor = BlameDetector(model_path="Lundsfryd/BlameBERT", batch_size=64)

# --- 1. Load the JSONL data ---
input_path = root_dir / "in" /"data"
output_dir = root_dir / "out" / "overlapping_blame" 
output_dir.mkdir(parents=True, exist_ok=True)

def run_source(predictor, input_path, source_name, n_rows=None):
    df = pd.read_json(input_path, lines=True, dtype={"date": str}, nrows=n_rows)
    df["date"] = pd.to_datetime(df["date"])          # real datetimes, not strings
    df["source"] = source_name

    texts = df["text"].astype(str).tolist()
    preds, confs = predictor.run_batch_prediction(texts, show_progress=True)
    assert len(preds) == len(df)

    df["predicted_class"] = preds
    df["confidence"] = confs

    # Keep the date as ISO text on disk so it round-trips cleanly
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_json(output_dir / f"{source_name}_predictions.jsonl",
                orient="records", lines=True, force_ascii=False)
    return df

df_a = run_source(predictor, Path(input_path, "segmented_ft_meetings.jsonl"), "ft_meetings", n_rows=N_ROWS)
df_b = run_source(predictor, Path(input_path, "segmented_parlspeech_meetings.jsonl"), "parlspeech_meetings", n_rows=N_ROWS)

combined = pd.concat([df_a, df_b], ignore_index=True)

out = combined.copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out.to_json(output_dir / "combined_predictions.jsonl",
            orient="records", lines=True, force_ascii=False)