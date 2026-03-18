import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class BlameDetector(object):

    def __init__(self, model_path, max_length=512, batch_size=32, model_from_path = True):

        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_from_path = model_from_path

        self.model_initialization()

        return

    def model_initialization(self):
        if self.model_from_path == True:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                device_map='auto'
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        elif self.model_from_path == False:
            self.model = self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")

        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        print(f"Model loaded successfully on {self.device}")

        return

    def predict(self, text):
        """Make a prediction on a single text input."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
       
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()

    def predict_batch(self, texts):
        """Make predictions on a batch of texts."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = probabilities[range(len(predicted_classes)), predicted_classes].cpu().numpy()
        
        return predicted_classes, confidences, probabilities.cpu().numpy()

    def run_prediction(self, text):
        """Single text prediction (backward compatibility)."""
        predicted_class, confidence, probs = self.predict(text)
        return predicted_class, confidence

    def run_batch_prediction(self, texts, show_progress=True):
        """
        Run predictions on a list of texts with batching and progress bar.
        
        Args:
            texts: List of text strings to predict on
            show_progress: Whether to show progress bar (default: True)
            
        Returns:
            predicted_classes: numpy array of predicted class indices
            confidences: numpy array of confidence scores
        """
        all_predictions = []
        all_confidences = []
        
        if show_progress:
            pbar = tqdm(total=len(texts), desc="Processing texts", unit="text")
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            predicted_classes, confidences, _ = self.predict_batch(batch_texts)
            
            all_predictions.extend(predicted_classes)
            all_confidences.extend(confidences)
            
            if show_progress:
                pbar.update(len(batch_texts))
        
        if show_progress:
            pbar.close()
        
        return np.array(all_predictions), np.array(all_confidences)

    def _load_jsonl(self, jsonl_path):
        """
        Load a JSONL file and return a list of dictionaries.
        
        Args:
            jsonl_path: Path to the JSONL file
            
        Returns:
            List of dictionaries, one per line
        """
        import json

        items = []
        print(f"Loading data from {jsonl_path}...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # skip empty lines
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")

        print(f"Loaded {len(items)} items from JSONL file.")
        return items

    def _save_jsonl(self, items, output_path):
        """
        Save a list of dictionaries to a JSONL file.
        
        Args:
            items: List of dictionaries to save
            output_path: Path to the output JSONL file
        """
        import json

        print(f"Saving results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Done! Saved {len(items)} items.")

    def predict_from_jsonl(self, jsonl_path, text_key='text', output_key='prediction',
                           confidence_key='confidence', show_progress=True):
        """
        Load JSONL, predict, and return results with predictions added.
        
        Args:
            jsonl_path: Path to JSONL file (one JSON object per line)
            text_key: Key in each JSON object containing the text to classify
            output_key: Key name for storing predictions (default: 'prediction')
            confidence_key: Key name for storing confidence scores (default: 'confidence')
            show_progress: Whether to show progress bar
            
        Returns:
            List of dictionaries with predictions added
        """
        items = self._load_jsonl(jsonl_path)

        print(f"Extracting texts from key '{text_key}'...")
        texts = [item[text_key] for item in items]

        print("Starting batch prediction...")
        predictions, confidences = self.run_batch_prediction(texts, show_progress=show_progress)

        for item, pred, conf in zip(items, predictions, confidences):
            item[output_key] = int(pred)
            item[confidence_key] = float(conf)

        return items

    def predict_from_jsonl_to_file(self, input_path, output_path, text_key='text',
                                   output_key='prediction', confidence_key='confidence',
                                   show_progress=True):
        """
        Load JSONL, predict, and save results to a new JSONL file.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to save output JSONL file
            text_key: Key containing text to classify
            output_key: Key name for predictions
            confidence_key: Key name for confidence scores
            show_progress: Whether to show progress bar
        """
        results = self.predict_from_jsonl(
            input_path, text_key, output_key, confidence_key, show_progress
        )

        self._save_jsonl(results, output_path)

        return results
    

    def compute_metrics(self, eval_pred):

        predictions, labels = eval_pred
        
        # Wrapping all metrics to floats for json serialization during model eval
        return {
            'recall': float(recall_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions)), #OBS CHANGE THIS
            'accuracy': float(accuracy_score(labels, predictions)), # Need rounding for these two computations (integer required)
            'f1': float(f1_score(labels, predictions, average='macro')), # macro f1 is better for imbalanced dataset
            'number_of_true_preds': sum(predictions),
            'number_of_true_labels': sum(labels)
        }
    
    def run_validation(self, validation_path, report_output_path = None):

        result = self.predict_from_jsonl(validation_path)

        true_labels = [entry["label"] for entry in result]
        predictions = [entry["prediction"] for entry in result]

        report = classification_report(true_labels, predictions)
        
        print(report)

        

        if report_output_path is not None:
            with open(report_output_path, "a") as f:  # "a" to append, not overwrite
                f.write("\n=== Validation Set Report ===\n\n")
                f.write(report)
            print(f"Validation report appended to {report_output_path}")

        evaluation = (predictions, true_labels)

        return self.compute_metrics(evaluation)