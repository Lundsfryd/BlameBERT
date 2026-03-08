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

    def normalize_evaluation(self, jsonl_path):
        """
        Read a JSONL file and convert 'evaluation' to binary:
        Blame -> 1
        no_blame -> 0
        """
        normalized = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                eval_str = item.get("evaluation", "").strip().lower()

                if eval_str == "blame":
                    item["evaluation"] = 1
                elif eval_str in {"no blame", "no_blame", "noblame"}:
                    item["evaluation"] = 0
                else:
                    raise ValueError(f"Unknown evaluation label: {item.get('evaluation')}")

                normalized.append(item)

        return normalized

    def align_by_paragraph_sentence(self, ann_a, ann_b):
        """
        Align two annotation lists by (paragraph_nr, sentence_nr).
        Returns list of (a, b) tuples where both annotators annotated the same unit.
        """

        def make_key(item):
            return (item["paragraph_nr"], item["sentence_nr"])

        map_a = {make_key(item): item for item in ann_a}
        map_b = {make_key(item): item for item in ann_b}

        common_keys = sorted(map_a.keys() & map_b.keys())

        aligned = [(map_a[k], map_b[k]) for k in common_keys]

        return aligned


    def cohens_kappa_from_pairs(self, pairs):
        """
        Compute Cohen's kappa from aligned annotation pairs.
        """
        if not pairs:
            raise ValueError("No overlapping annotations to compute agreement")

        a_labels = [a["evaluation"] for a, _ in pairs]
        b_labels = [b["evaluation"] for _, b in pairs]

        n = len(a_labels)

        po = sum(a == b for a, b in zip(a_labels, b_labels)) / n

        p_a_1 = sum(a_labels) / n
        p_a_0 = 1 - p_a_1

        p_b_1 = sum(b_labels) / n
        p_b_0 = 1 - p_b_1

        pe = (p_a_1 * p_b_1) + (p_a_0 * p_b_0)

        if pe == 1:
            return 1.0

        return (po - pe) / (1 - pe)


    def write_agreed_annotations(self, pairs, output_path):
        """
        Write only items where annotators agree.
        """
        with open(output_path, "w", encoding="utf-8") as out:
            for a, b in pairs:
                if a["evaluation"] == b["evaluation"]:
                    out.write(json.dumps(a, ensure_ascii=False) + "\n")


    def compare_LS_files(self, input_file_1, input_file_2, output_file):
        """
        Full pipeline:
        - Normalize evaluations
        - Align by (paragraph_nr, sentence_nr)
        - Compute Cohen's kappa
        - Write agreed annotations
        """

        ann_1 = self.normalize_evaluation(input_file_1)
        ann_2 = self.normalize_evaluation(input_file_2)

        aligned_pairs = self.align_by_paragraph_sentence(ann_1, ann_2)

        kappa = self.cohens_kappa_from_pairs(aligned_pairs)
        print(f"Cohen's κ (aligned): {kappa:.3f}")
        print(f"Aligned items: {len(aligned_pairs)}")

        self.write_agreed_annotations(aligned_pairs, output_file)

        print(f"Agreed annotations written to: {output_file}")