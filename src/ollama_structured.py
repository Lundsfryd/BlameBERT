import json
import tqdm
import pandas as pd
from pathlib import Path
from ollama import chat
from pydantic import BaseModel, ValidationError
from typing import List, Optional

class Blame(BaseModel):
    anklage: bool
    
def read_jsonl(file_path):
    """Read a jsonl file and return a list of records."""
    print(f'\n\n#### Reading "{file_path}" ####')
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                
    return records

def load_data(input_path):
    data_records = read_jsonl(input_path)
    data = pd.DataFrame(data_records)
    data = data[["text"]]

    return data



# %%
def runner(data, out_dir, retries=2):
    
    results = []

    for idx, sentence in enumerate(tqdm.tqdm(data, desc="Blame detection")):
        parsed = None
        
        for attempt in range(retries):
            try:
                response = chat(
                    model='qwen3.5:9b',
                    messages=[
                        {
                            'role': 'system',
                            'content': 'Du er en ekspert i at identificere hvornår politikere anklager hinanden for at være skyld i et negativt udfald. Identificér om der er nogle der anklager hinanden i sætningen. /no_think'
                        },
                        {
                            'role': 'user',
                            'content': f'{sentence}'
                        }
                    ],
                    format=Blame.model_json_schema(),
                    options={'temperature': 0}
                )
                
                raw = response.message.content
                
                if not raw or not raw.strip():
                    raise ValueError(f"Empty response on attempt {attempt + 1}")
                
                parsed = Blame.model_validate_json(raw)
                break
                
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                print(f"\n[Attempt {attempt + 1}] Parse error for sentence: '{sentence[:60]}...'\n  Error: {e}")
                if attempt == retries - 1:
                    print(f"  → Skipping after {retries} failed attempts.")
                    parsed = None
        
        results.append({
            'idx': idx,
            'sentence': sentence,
            'anklage': parsed.anklage if parsed else None
        })

    with open(f'{out_dir}/blame_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {out_dir}/blame_results.json")

def main():
    data = load_data(data_dir)
    runner(data, out_dir, retries=2)

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data" / "training_data" / "validation_set" / "validation_set.jsonl"
    out_dir = root_dir / "data" / "training_data" / "validation_set"
    main()