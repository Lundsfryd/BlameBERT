#import modules
import json
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


class TemplateManipulation(object):

    """
    This class serves two purposes after PBD has been implemented.

    1) It should create a number of datasets equivalent to the number 
        of hypothesis templates passed for evaluation of differentiated 
        agreement, i.e. at least five templates agreee on entailment to 
        classify as such, at least four templates agree on entailment... 
        and so on.

    2) Compute hypothesis-specific stats regarding amount of labels as true.
    
    
    """

    # --------------------------------------------------------------------#

    def __init__(self, input_path, output_path):

        self.input_path = input_path
        self.output_path = output_path
        self.outdir = os.path.dirname(output_path)

        return
    
    
    # --------------------------------------------------------------------#

    def detect_n_hypotheses(self):
        """Detect number of Hyp_*_blame fields from the first non-empty line."""
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    first_record = json.loads(line)
                    self.n_hyp = sum(
                        1 for k in first_record
                        if k.startswith("Hyp_") and k.endswith("_blame")
                    )
                    break
            else:
                raise ValueError("Input file is empty")
            
            print(f"number of hypothesis templates detected: {self.n_hyp}...\n")
            return self.n_hyp
        
    # --------------------------------------------------------------------#
    
    def add_n_hyp_entail(self):

        self.blame_keys = [f"Hyp_{i}_blame" for i in range(1, self.n_hyp + 1)]

        with open(self.input_path, "r", encoding="utf-8") as fin, \
            open(self.output_path, "w", encoding="utf-8") as fout:


            for line in tqdm(fin, desc= f"Adding key 'n_hyp entail' to: {self.output_path}..."):
                if not line.strip():
                    continue

                record = json.loads(line)

                record["n_hyp_entail"] = sum(
                    record.get(k, 0) for k in self.blame_keys
                )

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # --------------------------------------------------------------------#

    def setup_writers(self):

        writers = {}
        print("Setting up writers for labeling by level of agreement...\n")
        for threshold in range(1, self.n_hyp + 1):
            fname = (
                f"{threshold}_agreement.jsonl"
                if threshold == self.n_hyp
                else f"{threshold}_{self.n_hyp}_agreement.jsonl"
            )
            writers[threshold] = open(
                os.path.join(self.hyp_agree_outdir, fname),
                "w",
                encoding="utf-8"
            )
        return writers
    
    # --------------------------------------------------------------------#

    def label_writer(self, writers):

        try:
            with open(self.output_path, "r", encoding="utf-8") as fin:
                for line in tqdm(fin, desc= f"Reading input and writing output files for threshold of agreement..."):
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    
                    if "n_hyp_entail" not in record:
                        raise KeyError("Missing 'n_hyp_entail' in record")
                    
                    entail = record["n_hyp_entail"]

                    for threshold, fout in writers.items():
                        record_out = record.copy()
                        record_out["label"] = int(entail >= threshold)
                        fout.write(json.dumps(record_out, ensure_ascii=False) + "\n")

        finally:
            for f in writers.values():
                f.close()
        
    # --------------------------------------------------------------------#
    
    def create_dir(self):


        # if the directory is not present 
        # then create it.
        self.hyp_agree_outdir = os.path.join(self.outdir,
                                     "test_diff_hypothesis_agreemen")
        
        print(f"Making output directory for datasets of different levels of agreement:\n{self.hyp_agree_outdir}\n")

        os.makedirs(self.hyp_agree_outdir, exist_ok=True)
        return
    
    # --------------------------------------------------------------------#

    def create_datasets(self):

        self.create_dir()

        # Open all output files at once
        writers = self.setup_writers()

        #write to those files
        self.label_writer(writers)

        return
    
    # --------------------------------------------------------------------#
    
    def create_datasets_diff_agreement(self):

        #detect n hypothesis entries in data
        self.detect_n_hypotheses()

        #add a key denoting how many hypothesis labelled true
        self.add_n_hyp_entail()

        #create datasets of different levels of agreement
        self.create_datasets()


        return
    
    # --------------------------------------------------------------------#

    def compute_agreement_statistics_per_hyp(self):
        """
        Compute per-threshold agreement statistics and per-hypothesis percentages.
        """
        counts_threshold = Counter()  # for n_hyp_entail thresholds
        counts_hyp = Counter()        # for each individual hypothesis
        total = 0

        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc= "Reading lines of output file for statistics..."):
                if not line.strip():
                    continue

                record = json.loads(line)
                entail = record.get("n_hyp_entail", 0)

                total += 1

                # per-threshold counts
                for k in range(1, self.n_hyp + 1):
                    if entail >= k:
                        counts_threshold[k] += 1

                # per-hyp counts
                for key in self.blame_keys:
                    if record.get(key, 0) == 1:
                        counts_hyp[key] += 1

        # compute percentages
        pct_threshold = {
            k: 100 * counts_threshold[k] / total for k in range(1, self.n_hyp + 1)
        }

        pct_hyp = {
            key: 100 * counts_hyp[key] / total for key in self.blame_keys
        }

        return pct_threshold, pct_hyp, total
    

    def print_agreement_statistics_per_hyp(self, pct_threshold, pct_hyp, total):
        print(f"\nTotal samples: {total:,}\n")

        print("=== Per-threshold agreement percentages ===")
        for k, pct in pct_threshold.items():
            print(f"Agreement ≥ {k} hypotheses: {pct:.2f}%")

        print("\n=== Per-hypothesis percentages ===")
        for key, pct in pct_hyp.items():
            print(f"{key}: {pct:.2f}%")



    def plot_agreement_statistics_per_hyp(self, pct_threshold, pct_hyp):
        
        # per-threshold bar plot
        plt.figure(figsize=(8, 4))
        plt.bar(pct_threshold.keys(), pct_threshold.values())
        plt.xlabel("Minimum hypothesis agreement")
        plt.ylabel("Percentage of samples (%)")
        plt.title("Agreement rate by threshold")
        plt.xticks(list(pct_threshold.keys()))
        plt.ylim(0, self.ylim)
        plt.show()

        # per-hypothesis bar plot
        plt.figure(figsize=(10, 4))
        plt.bar(pct_hyp.keys(), pct_hyp.values())
        plt.xlabel("Hypothesis key")
        plt.ylabel("Percentage of samples labeled 1 (%)")
        plt.title("Percentage of 1s per hypothesis")
        plt.ylim(0, self.ylim)
        plt.show()

    def run_statistics(self, ylim = 10):

        self.ylim = ylim

        pct_threshold, pct_hyp, total = self.compute_agreement_statistics_per_hyp()
        self.print_agreement_statistics_per_hyp(pct_threshold, pct_hyp, total)
        self.plot_agreement_statistics_per_hyp(pct_threshold, pct_hyp)

        return
    


def main():

    parser = argparse.ArgumentParser(
                    prog='Template manipulation',
                    description='Creates templates based on level of agreement. Also prodices template statistics')
    
    parser.add_argument("--input_path", 
                        type=Path,
                        required=True) # add argument
    
    parser.add_argument("--output_path", 
                        type=Path,
                        required=True) # add argument
    
    parser.add_argument("--ylim", 
                        type=int,
                        default=10) # add argument


    args = parser.parse_args()

    TM = TemplateManipulation(input_path = args.input_path,
                            output_path = args.output_path)

    TM.create_datasets_diff_agreement()
    TM.run_statistics(ylim=args.ylim)

    TM.create_datasets_diff_agreement()
    TM.run_statistics()



if __name__ == "__main__":

    main()


    

    