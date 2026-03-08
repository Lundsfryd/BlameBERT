# import modules
import json
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, recall_score, confusion_matrix

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


    def json_to_jsonl(self, input_path, output_path):
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
                    a["label"] = a.get("evaluation")
                    a.pop("evaluation",0)
                    out.write(json.dumps(a, ensure_ascii=False) + "\n")



    def agreement_stats_from_pairs(self, pairs):
        """
        Compute per-label agreement statistics from aligned annotation pairs.
        Returns a dict with stats for label=1 (Blame), label=0 (No Blame), and overall.
        """
        if not pairs:
            raise ValueError("No overlapping annotations to compute agreement")

        a_labels = [a["evaluation"] for a, _ in pairs]
        b_labels = [b["evaluation"] for _, b in pairs]
        n = len(a_labels)

        # --- Overall ---
        overall_agree = sum(a == b for a, b in zip(a_labels, b_labels))

        # --- Label = 1 (Blame) ---
        # Both said Blame
        both_1       = sum(a == 1 and b == 1 for a, b in zip(a_labels, b_labels))
        # At least one said Blame (union, for denominator)
        either_1     = sum(a == 1 or  b == 1 for a, b in zip(a_labels, b_labels))
        # Only annotator A said Blame
        only_a_1     = sum(a == 1 and b == 0 for a, b in zip(a_labels, b_labels))
        # Only annotator B said Blame
        only_b_1     = sum(a == 0 and b == 1 for a, b in zip(a_labels, b_labels))

        # --- Label = 0 (No Blame) ---
        both_0       = sum(a == 0 and b == 0 for a, b in zip(a_labels, b_labels))
        either_0     = sum(a == 0 or  b == 0 for a, b in zip(a_labels, b_labels))
        only_a_0     = sum(a == 0 and b == 1 for a, b in zip(a_labels, b_labels))
        only_b_0     = sum(a == 1 and b == 0 for a, b in zip(a_labels, b_labels))

        def safe_div(num, denom):
            return num / denom if denom else None

        return {
            "n_total": n,
            "n_agree": overall_agree,
            "overall_agreement_pct": safe_div(overall_agree, n),

            "blame": {
                "both_annotated":    both_1,
                "only_annotator_1":  only_a_1,
                "only_annotator_2":  only_b_1,
                "union":             either_1,
                # What fraction of all Blame instances (union) did both agree on?
                "agreement_pct":     safe_div(both_1, either_1),
                # Of all sentences, how often did BOTH say Blame?
                "joint_rate":        safe_div(both_1, n),
            },

            "no_blame": {
                "both_annotated":    both_0,
                "only_annotator_1":  only_a_0,
                "only_annotator_2":  only_b_0,
                "union":             either_0,
                "agreement_pct":     safe_div(both_0, either_0),
                "joint_rate":        safe_div(both_0, n),
            },
        }


    def allignment_stats(self, pairs):

        y_model = []
        y_ann = []

        for a, b in pairs:
            # Safety: ensure same sentence unit
            if (a["paragraph_nr"], a["sentence_nr"]) != (b["paragraph_nr"], b["sentence_nr"]):
                continue

            # Only include pairs where annotators agree on evaluation
            if a["evaluation"] != b["evaluation"]:
                continue

            evaluation = int(a["evaluation"])  # same in both since they agree
            label = a.get("label") 

            y_ann.append(evaluation)
            y_model.append(label)
        
        cm = confusion_matrix(y_ann, y_model)

        stats = {
            'recall': float(recall_score(y_ann, y_model)),
            'precision': float(average_precision_score(y_ann, y_model)),
            'accuracy': float(accuracy_score(y_ann, y_model)), # Need rounding for these two computations (integer required)
            'f1': float(f1_score(y_ann, y_model, average='macro'))} # macro f1 is better for imbalanced dataset


        return stats, cm

        


    def compare_LS_files(self, input_file_1, input_file_2, output_file):
        """
        Full pipeline:
        - Normalize evaluations
        - Align by (paragraph_nr, sentence_nr)
        - Compute Cohen's kappa
        - Compute per-label agreement statistics
        - Write agreed annotations
        """

        ann_1 = self.normalize_evaluation(input_file_1)
        ann_2 = self.normalize_evaluation(input_file_2)

        aligned_pairs = self.align_by_paragraph_sentence(ann_1, ann_2)

        kappa = self.cohens_kappa_from_pairs(aligned_pairs)
        stats = self.agreement_stats_from_pairs(aligned_pairs)
        allign_stats, con_mat = self.allignment_stats(aligned_pairs)

        # ── Print summary ──────────────────────────────────────────────
        print("=" * 50)
        print("INTER-ANNOTATOR AGREEMENT REPORT")
        print("=" * 50)

        print(f"\nAligned sentence pairs : {stats['n_total']}")
        print(f"Overall agreed pairs   : {stats['n_agree']}  "
            f"({stats['overall_agreement_pct']:.1%})")
        print(f"Cohen's κ              : {kappa:.3f}")

        for label_name, label_key in [("BLAME  (label=1)", "blame"),
                                    ("NO BLAME (label=0)", "no_blame")]:
            s = stats[label_key]
            print(f"\n  {label_name}")
            print(f"    Both annotated      : {s['both_annotated']}")
            print(f"    Only annotator 1    : {s['only_annotator_1']}")
            print(f"    Only annotator 2    : {s['only_annotator_2']}")
            if s["agreement_pct"] is not None:
                print(f"    Agreement %         : {s['agreement_pct']:.1%}  "
                    f"(agreed / union = {s['both_annotated']}/{s['union']})")
            else:
                print(f"    Agreement %         : N/A (neither annotator used this label)")

        print("=" * 50)

        print(f"\nConfusion matrix for alignment between aggreed upon annotated labels (y_true)\nand label deceided by DEBATE model + heuristics + level of agreeement:\n{con_mat} \n")
        print(f"\nAllignment statsbetween annotater agreement and prelim blame localization:\n{allign_stats}")

        print("=" * 50)



        self.write_agreed_annotations(aligned_pairs, output_file)
        print(f"\nAgreed annotations written to: {output_file}")

        return {"kappa": kappa, **stats}