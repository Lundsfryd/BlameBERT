# import modules
import json


class Formatter(object):


    def __init___(self):

        return

    def convert_to_label_studio(self, input_path, output_path):
        """
        Convert jsonl file to Label Studio import format.
        
        Args:
            input_path: path to input jsonl file
            output_path: path to output json file
        """
        tasks = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                task = {
                    "data": {
                        "text": record["text"],
                        "meta": {k: v for k, v in record.items() if k != "text"}
                    }
                }
                tasks.append(task)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"Converted {len(tasks)} records to Label Studio format: {output_path}")


    def json_to_jsonl(input_path, output_path):
        """
        Convert a Label Studio-style JSON file to a flattened JSONL file.
        """

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(output_path, "w", encoding="utf-8") as out:
            for item in data:
                meta = item.get("meta", {})

                jsonl_item = {
                    "date": meta.get("date"),
                    "speaker": meta.get("speaker"),
                    "party": meta.get("party"),
                    "paragraph_nr": meta.get("paragraph_nr"),
                    "sentence_nr": meta.get("sentence_nr"),
                    "partyfacys_ID": meta.get("partyfacys_ID"),
                    "text": item.get("text"),
                    "label": meta.get("label"),
                    "evaluation": item.get("evaluation"),
                }

                out.write(json.dumps(jsonl_item, ensure_ascii=False) + "\n")