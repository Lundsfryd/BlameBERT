#import modules
import json


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

        self.get_params()

        return
    
    # --------------------------------------------------------------------#

    def get_params(self):
        pass

        return

    
    # --------------------------------------------------------------------#

    def detect_n_hypotheses(self):
        """Detect number of Hyp_*_blame fields from the first non-empty line."""
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    first_record = json.loads(line)
                    n_hyp = sum(
                        1 for k in first_record
                        if k.startswith("Hyp_") and k.endswith("_blame")
                    )
                    break
            else:
                raise ValueError("Input file is empty")
            
            return n_hyp
    
    def add_n_hyp_entail(self, n_hyp):

        blame_keys = [f"Hyp_{i}_blame" for i in range(1, n_hyp + 1)]

        with open(self.input_path, "r", encoding="utf-8") as fin, \
            open(self.output_path, "w", encoding="utf-8") as fout:


            for line in fin:
                if not line.strip():
                    continue

                record = json.loads(line)

                record["n_hyp_entail"] = sum(
                    record.get(k, 0) for k in blame_keys
                )

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # --------------------------------------------------------------------#

    def create_datasets(self):

        pass
    
    def create_datasets_diff_agreement(self):

        n_hyp = self.detect_n_hypotheses()
        self.add_n_hyp_entail(n_hyp)

        self.create_datasets()


        return
    
    # --------------------------------------------------------------------#

    def run_statistics(self):



        return