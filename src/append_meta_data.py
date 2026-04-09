import pandas as pd
import json
import ast

# Load government data once
regerings_data = pd.read_csv("/work/MarkusLundsfrydJensen#1865/Bachelor_project/data/raw_data/danish_govs.csv")

regerings_data["Start Date"] = pd.to_datetime(regerings_data["Start Date"])
regerings_data["End Date"]   = pd.to_datetime(regerings_data["End Date"])


def find_government(date):
    """Return list of party letters in government at given date."""
    match = regerings_data[
        (regerings_data["Start Date"] <= date) &
        (regerings_data["End Date"] >= date)
    ]
    
    if not match.empty:
        parties = match['Party Letter'].iloc[0]
        return ast.literal_eval(parties)  # convert string → list
    else:
        return None


def add_in_gov_to_jsonl(input_path, output_path):
    """Read jsonl, append 'in_gov', and write new jsonl."""
    
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            entry = json.loads(line)
            
            # Parse date
            try:
                date = pd.to_datetime(entry["date"])
            except:
                entry["in_gov"] = None
                outfile.write(json.dumps(entry) + "\n")
                continue
            
            party = entry.get("party", None)
            parties_gov = find_government(date)
            
            if parties_gov is None or party is None:
                entry["in_gov"] = None
            else:
                entry["in_gov"] = int(party in parties_gov)
            
            outfile.write(json.dumps(entry) + "\n")


def jsonl_to_csv(jsonl_path, csv_path):
    """Convert jsonl file to csv."""
    
    data = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)



# Step 1: Add in_gov
add_in_gov_to_jsonl(
    "/work/MarkusLundsfrydJensen#1865/Bachelor_project/data/inference/predicted_inference_data.jsonl",
    "/work/MarkusLundsfrydJensen#1865/Bachelor_project/data/inference/predicted_inference_data_with_gov.jsonl"
)

# Step 2: Convert to CSV
jsonl_to_csv(
    "/work/MarkusLundsfrydJensen#1865/Bachelor_project/data/inference/predicted_inference_data_with_gov.jsonl",
    "/work/MarkusLundsfrydJensen#1865/Bachelor_project/data/inference/predicted_inference_data_with_gov.csv"
)