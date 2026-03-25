import os
import torch
import itertools
import json
from pathlib import Path
from training_setup import model_trainer
from blame_detection import BlameDetector
import gc

# ------------------------------------------------------------------- #

parent_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(parent_dir)
data_dir = os.path.join(root_dir, "data", "training_data", "diff_hypothesis_agreemen")

validation_data_path = os.path.join(os.path.dirname(data_dir),
                                    "validation_set",
                                    "validation_set.jsonl")

output_dir = os.path.join(root_dir, "training_output")

datasets = [
    {"path": Path(os.path.join(data_dir, "1_5_agreement.jsonl")), "model_name": "data_1_5"},
    {"path": Path(os.path.join(data_dir, "2_5_agreement.jsonl")), "model_name": "data_2_5"},
    {"path": Path(os.path.join(data_dir, "3_5_agreement.jsonl")), "model_name": "data_3_5"},
    {"path": Path(os.path.join(data_dir, "4_5_agreement.jsonl")), "model_name": "data_4_5"},
    {"path": Path(os.path.join(data_dir, "5_agreement.jsonl")),   "model_name": "data_5"},
]

# ── Hyperparameter grid ──────────────────────────────────────────────
LEARNING_RATES   = [1e-6, 1e-5, 1e-4]
LR_SCHEDULERS    = ["linear"]
ALPHA_MODES      = ["sqrt", "two_thirds", "raw"]   # applied inside model_trainer via load_data
# --------------------------------------------------------------------
# NOTE: alpha_mode is passed to model_trainer and forwarded to load_data,
# which computes the actual tensor. The three modes correspond to:
#   sqrt:       torch.sqrt(raw_weights) / min   → softest
#   two_thirds: raw_weights ** (2/3)   / min   → medium
#   raw:        raw_weights            / min   → strongest
# --------------------------------------------------------------------

# Track best result per dataset for final BlameDetector evaluation
best_per_dataset = {}   # ds_model_name -> {"mcc": float, "model": ..., "run_name": str}

sweep = list(itertools.product(datasets, LEARNING_RATES, LR_SCHEDULERS, ALPHA_MODES))
total = len(sweep)

print(f"\n{'='*60}")
print(f"  Starting sweep: {total} runs")
print(f"  ({len(datasets)} datasets × {len(LEARNING_RATES)} LRs × "
      f"{len(LR_SCHEDULERS)} schedulers × {len(ALPHA_MODES)} alpha modes)")
print(f"{'='*60}\n")

results_log = []   # collect all run summaries for inspection after sweep

for run_idx, (ds, lr, scheduler, alpha_mode) in enumerate(sweep, 1):

    run_name = (f"{ds['model_name']}"
                f"__lr{lr}"
                f"__sched-{scheduler}"
                f"__alpha-{alpha_mode}")

    print(f"\n[{run_idx}/{total}] {run_name}")

    report_path = os.path.join(output_dir, f"report__{run_name}.txt")

    try:
        model, best_mcc = model_trainer(
            data_input_path=ds["path"],
            output_dir=output_dir,
            model_name=run_name,
            save_model=True,
            subset=20000,
            report_path=report_path,
            batch_size=256,
            learning_rate=lr,
            lr_scheduler=scheduler,      # new arg — see model_trainer changes below
            alpha_mode=alpha_mode,       # new arg — see load_data changes below
        )

    except Exception as e:
        print(f"  [ERROR] Run failed: {e}")
        results_log.append({"run": run_name, "mcc": None, "error": str(e)})
        continue

    results_log.append({"run": run_name, "mcc": best_mcc})
    print(f"  → best val MCC: {best_mcc:.4f}")

    # Track best model per dataset
    ds_key = ds["model_name"]
    if ds_key not in best_per_dataset or best_mcc > best_per_dataset[ds_key]["mcc"]:
        best_per_dataset[ds_key] = {
            "mcc":      best_mcc,
            "model":    model,
            "run_name": run_name,
        }
    else:
        # Not the best — free memory immediately
        del model
        torch.cuda.empty_cache()
        gc.collect()

# ── Final evaluation: run BlameDetector on best model per dataset ────
print(f"\n{'='*60}")
print("  Sweep complete. Running BlameDetector on best models.")
print(f"{'='*60}\n")

for ds_key, best in best_per_dataset.items():
    print(f"\n[BlameDetector] {ds_key}  (run: {best['run_name']},  MCC: {best['mcc']:.4f})")

    report_path = os.path.join(output_dir, f"blame_detector_report__{best['run_name']}.txt")

    try:
        detector = BlameDetector(
            model_path=best["model"],
            max_length=512,
            batch_size=256,
            model_from_path=False,
        )
        detector.run_validation(validation_data_path, report_path)
        del detector

    except Exception as e:
        print(f"  [ERROR] BlameDetector failed for {ds_key}: {e}")

    del best["model"]
    torch.cuda.empty_cache()
    gc.collect()

# ── Save full results log ────────────────────────────────────────────
log_path = os.path.join(output_dir, "sweep_results.json")
with open(log_path, "w") as f:
    json.dump(results_log, f, indent=2)

print(f"\nAll results saved to {log_path}")