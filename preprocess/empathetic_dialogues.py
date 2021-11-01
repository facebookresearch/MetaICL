import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class EmpatheticDialogues(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "empathetic_dialogues"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_train_test_lines(self, dataset):

        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")

        np.random.seed(42)
        np.random.shuffle(test_lines)
        n = len(test_lines)
        test_lines = test_lines[:int(0.2*n)]
        # using 20% of test cases, otherwise it's too slow to do evaluation

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = datapoint["prompt"].replace("_comma_", ",") + " [SEP] " + datapoint["context"]
            
            if "hit:" in datapoint["utterance"]:
                continue # some bad lines
            lines.append((input_text, datapoint["utterance"].replace("_comma_", ",").replace("\n", " ").replace("\t", " ").replace("\r", " ")))

        # merge same prompts
        d = {}
        for line in lines:
            if line[0] in d:
                d[line[0]].append(line[1])
            else:
                d[line[0]] = [line[1]]
        
        lines = []
        for k, v in d.items():
            lines.append((k, "\t".join(v)))

        return lines

    def load_dataset(self):
        return datasets.load_dataset('empathetic_dialogues')

def main():
    dataset = EmpatheticDialogues()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()