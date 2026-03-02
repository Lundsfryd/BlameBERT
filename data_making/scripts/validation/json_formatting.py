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