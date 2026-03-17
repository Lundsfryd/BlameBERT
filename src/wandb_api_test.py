import wandb
import pandas as pd
from pathlib import Path
import json

"""
API call to download all tables from wandb
OBS: should take run as argument, in order not to overlap
"""

run = None #eg 68nf8w58

api = wandb.Api()
run = api.run(f"markuslundsfryd-aarhus-university/mmbert-danish-politics/{run}")

output_dir = Path("/work/MarkusLundsfrydJensen#1865/data_outside_git/embedding_data")
output_dir.mkdir(exist_ok=True)

for f in run.files():
    if "layer_embeddings_table" in f.name and f.name.endswith(".json"):
        print(f"Downloading {f.name} ...")
        f.download(root=str(output_dir), replace=True)

# Now convert all downloaded json tables to csv
for json_file in output_dir.rglob("*.json"):
    with open(json_file) as fp:
        raw = json.load(fp)
    df = pd.DataFrame(raw["data"], columns=raw["columns"])
    csv_path = json_file.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")