#import modules
import argparse
import pandas as pd
import json
from tqdm import tqdm
import spacy
import dacy
import os
import json
from pathlib import Path


class SentenceSegmentation(object):

    def __init__(self, input_path, output_path):
        
        self.input_path = input_path

        self.output_path = output_path

        print("attempting to initialize spacy with gpu... ")
        print(spacy.prefer_gpu())
        spacy.prefer_gpu()

        print("loading 'da_dacy_large_trf'...") 
        self.nlp = spacy.load("da_dacy_large_trf", disable=["ner", "trainable_lemmatizer", "morphologizer", "coref", "entity_linker"]) #dacy load does not work, but spacy with the dacy corpus
        
        print(f"The loaded model name is: {self.nlp.meta['name']}") # shoudl say the above name

        #print(f"Confirming use of GPU. Value should be >0 {cupy.cuda.runtime.getDeviceCount()}")

        return
    


    def read_data(self):

        print("reading csv file...\n")

        self.df = pd.read_csv(self.input_path)

        return
    
    def extract_texts(self, n_rows = -1):

        print("extracting texts from csv file... \n")

        self.texts = self.df["text"][:n_rows].tolist()
        
        return
    
    def write_data(self, file, sentences):

        file.write(json.dumps(sentences, ensure_ascii=False) + "\n")


        return
    

    def run_sentences_within_paragraph(self, document_sentences, current_row, para):


        for i, sent in enumerate(document_sentences):

            sentence_data = {
                "date": current_row["date"],
                "speaker": current_row["speaker"],
                "party": current_row["party"],
                "paragraph_nr": para,
                "sentence_nr": i,
                "partyfacys_ID": current_row["partyfacts_ID"],
                "text": str(sent)
            }
            
            self.write_data(file = self.f, sentences = sentence_data)

        return

    
    def run_segmentation(self, batch_size, n_process):

        with self.output_path.open("w", encoding="utf-8") as self.f:

            print("initializing pipeline...\n")
            for paragraph, doc in enumerate(tqdm(self.nlp.pipe(self.texts, batch_size=batch_size, n_process=n_process), total=len(self.texts))):

                row = self.df.iloc[paragraph]

                self.run_sentences_within_paragraph(document_sentences = doc.sents, 
                                                    current_row = row, 
                                                    para = paragraph)
                
        return
    
    
    def sentence_segmentation(self, n_rows = -1, batch_size=2, n_process=1):

        self.read_data()
        self.extract_texts(n_rows=n_rows)

        print("Starting segmentation...\n")
        self.run_segmentation(batch_size=batch_size, n_process=n_process)

        return

    


def main():

    parser = argparse.ArgumentParser(
                    prog='SentenceSegmentation',
                    description='Takes input as list of paragraphs and returns segmented sentences and metadata i jsonl format')
    
    parser.add_argument("--inpath_to_csv", 
                        type=Path,
                        required=True) # add argument

    parser.add_argument("--outpath_to_jsonl", 
                        type=Path,
                        required=True) # add argument

    parser.add_argument("--n_rows", 
                        type=int,
                        default=-1, 
                        help="number of paragraphs to evaluate") # add argument
    
    parser.add_argument("--batch_size", 
                        type=int,
                        default=2, 
                        help="Batch size for nlp pipe") # add argument
    
    parser.add_argument("--n_process", 
                        type=int,
                        default=1, 
                        help="n_processes for nlp pipe") # add argument
    
    args = parser.parse_args()

    print(f"""\nRunning sentence segmentation on inpath_to_csv: {args.inpath_to_csv}\n 
          Outpath_to_jsonl: {args.outpath_to_jsonl}\n 
        n_rows = {args.n_rows}\n
         batch_size = {args.batch_size}\n
          n_process = {args.n_process}\n """)


    SS = SentenceSegmentation(input_path=args.inpath_to_csv, output_path=args.outpath_to_jsonl)

    SS.sentence_segmentation(n_rows = args.n_rows, batch_size= args.batch_size, n_process=args.n_process)

    print("Sentence segmentation complete")
    return



if __name__ == "__main__":

    main()

