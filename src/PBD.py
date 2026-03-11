#import modules
import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path
import os
import argparse


class DebateEntailment(object):

    def __init__(self, inpath, outpath, batch_size = 2):

        
        self.outpath = outpath
        self.batch_size = batch_size
        self.batch_index = 0  # tracks which record in self.records to start from for each batch

        self.read_jsonl(inpath)
        self.get_params()
        self.get_template_info()
        
        return
    
    def get_template_info(self):

        hypothesis_path = Path(os.path.join(".",
                           "hypothesis_templates.txt"))

        self.labels = ["blame", "praise", "neutral"]
        self.hypothesis_templates = self.read_file_lines(hypothesis_path)
        return
    
    def read_file_lines(self, filepath):
        with open(filepath, 'r') as f:
            return f.read().splitlines()
    
    

    def get_params(self):

        self.initialize_outpath()

        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Device: {'cuda' if self.device == 0 else 'cpu'}")
        print("remember to change to large. Currently base")
        # Use base model for speed (large is much slower)
        self.pipe = pipeline("zero-shot-classification", model='mlburnham/Political_DEBATE_base_v1.0', device = self.device, batch_size = self.batch_size)
        

        return
    
    def initialize_outpath(self):
        """Clear the output file if it already exists to prevent duplicate entries."""
        
        print("Initializing outpath...\n")
        
        with open(self.outpath, 'w', encoding='utf-8') as f:
            pass

        return

    def read_jsonl(self, file_path):

        """Read a jsonl file and return a list of records."""

        print("Reading input .jsonl file...\n")
        self.records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        return self.records
    

    def evaluate_premise(self, doc):

        return self.pipe(doc, self.labels, hypothesis_template=self.hypothesis_template, multi_label=True)

    def blame_in_batch(self, sentences):

        #results = []

        for i in tqdm(range(0, len(sentences), self.batch_size), desc = f"Applying blame detection with hypothesis nr: {self.hypothesis_number}"):
            batch = sentences[i:i+self.batch_size]
            output = self.evaluate_premise(batch)
            if isinstance(output, list):
                
                result = self.blame_by_heuristics(output)
                self.write_blame_to_jsonl(result)
                
            else:
                print("Instance is not list...\n")
        return #results
    
    def blame_by_heuristics(self, blame_list):

        """evaluate from list of dicts of 
        absolute probabilities of entailment to
        0's or 1's.
        
        Returns a binary list for blame per sentence:
        1 if blame is highest among labels and >= 0.8, else 0
        Handles arbitrary label order.
        """


        # List comprehension is faster than appending in a loop
        blame_binary = [
            int(
                (label_score := {label: score for label, score in zip(sent['labels'], sent['scores'])})['blame']
                >= max(label_score.get('praise', 0.0), label_score.get('neutral', 0.0), 0.8)
            )
            for sent in blame_list
        ]

        return blame_binary

    def write_blame_to_jsonl(self, decoded):
        """Append translated batch to output file, adding the 'hyp_ntemp_blame' field."""

        with open(self.outpath, self.write_mode, encoding='utf-8') as f:  # 'a' = append, not overwrite
            for blame_value in decoded:
                record = self.records[self.batch_index].copy()
                record[f'Hyp_{self.hypothesis_number}_blame'] = blame_value
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                self.batch_index += 1
            self.write_mode = 'a'  # after first write of this hypothesis, switch back to append
        return
    

    def run_hypothesis_entailment(self):

        texts = [record['translated_text'] for record in self.records] 

        for i, template in enumerate(self.hypothesis_templates):

            self.hypothesis_template = template

            print(f"now for this hypothesis:\n{self.hypothesis_template}")
            self.hypothesis_number = i+1
            print(f"this is self.hypothesis_number:\n{self.hypothesis_number}")
            self.batch_index = 0


            #after first run, append to already existing file instead
            if i > 0:
                self.read_jsonl(self.outpath)
                texts = [record['translated_text'] for record in self.records]  # refresh texts
                self.write_mode = 'w'  # overwrite with enriched records
            else:
                self.write_mode = 'a'

            self.blame_in_batch(texts)

        return
    


def main():

    parser = argparse.ArgumentParser(
                    prog='Prelimenary Blame Detection (PBD)',
                    description='Does PBD from jsonl "translated_text column" and evaluate entailment in comparison to found templates')
    
    parser.add_argument("--input_path_jsonl", 
                        type=Path,
                        required=True) # add argument
    
    parser.add_argument("--output_path_jsonl", 
                        type=Path,
                        required=True) # add argument
    
    parser.add_argument("--batch_size", 
                        type=int,
                        default=2) # add argument
    
    args = parser.parse_args()

    DE = DebateEntailment(inpath=args.input_path_jsonl,
                      outpath=args.output_path_jsonl,
                      batch_size=args.batch_size)

    DE.run_hypothesis_entailment()

    return


if __name__ == "__main__":

    main()