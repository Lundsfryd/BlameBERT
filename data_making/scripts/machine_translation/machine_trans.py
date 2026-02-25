#import modules
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.cuda.amp import autocast
import IProgress
import ipywidgets
import sentencepiece
import sacremoses


class MachineTranslation(object):


    def __init__(self, jsonl_path, outpath, batch_size=2, gpu=False):
        
        self.batch_size = batch_size
        self.outpath = outpath
        self.gpu = gpu
        self.batch_index = 0  # tracks which record in self.records to start from for each batch

        self.set_params()
        self.initialize_outpath()
        self.read_jsonl(jsonl_path)
        return
    

    def set_params(self):

        print("\nLoading model...\n")
        
        model_name = "Helsinki-NLP/opus-mt-da-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.device = "cuda" if self.gpu else "cpu"
        self.model.to(self.device)
        self.model.eval()

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
    

    def translate_text(self, texts):

        print("Translating texts...\n")

        for i in tqdm(range(0, len(texts), self.batch_size)):

            batch = texts[i:i+self.batch_size]

            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.device == "cuda":
                with torch.no_grad(), autocast():
                    translated = self.model.generate(**inputs)
            else:
                with torch.no_grad():
                    translated = self.model.generate(**inputs)

            decoded = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

            self.write_translated_jsonl(decoded)
        
        return
    

    def write_translated_jsonl(self, decoded):
        """Append translated batch to output file, substituting the 'text' field."""

        with open(self.outpath, 'a', encoding='utf-8') as f:  # 'a' = append, not overwrite
            for translated_text in decoded:
                record = self.records[self.batch_index].copy()
                record['text'] = translated_text
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                self.batch_index += 1
        return
    

    def run_translation(self):

        texts = [record['text'] for record in self.records] 

        self.translate_text(texts)






def main():

    parser = argparse.ArgumentParser(
                    prog='Machine Translation',
                    description='Does MT from jsonl "text column" file to jsonl file')
    
    parser.add_argument("--input_path_jsonl", 
                        type=Path,
                        required=True) # add argument
    
    parser.add_argument("--output_path_jsonl", 
                        type=Path,
                        required=True) # add argument
    
    parser.add_argument("--batch_size", 
                        type=int,
                        default=2) # add argument
    
    parser.add_argument("--GPU",
                        action="store_true",
                        help="Use GPU acceleration")
    
    args = parser.parse_args()

    MT = MachineTranslation(jsonl_path = args.input_path_jsonl, 
                        outpath = args.output_path_jsonl, 
                        batch_size=args.batch_size, 
                        gpu=args.GPU)

    MT.run_translation()

    return


if __name__ == "__main__":

    main()