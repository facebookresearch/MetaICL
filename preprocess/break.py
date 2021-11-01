import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Break(FewshotGymTextToTextDataset):

    def __init__(self, subset_identifier):
        self.hf_identifier = "break-" + subset_identifier 
        self.subset_identifier = subset_identifier
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("question: " + datapoint["question_text"], datapoint["decomposition"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('break_data', self.subset_identifier)

def main():

    for subset_identifier in ["QDMR", "QDMR-high-level"]:
        dataset = Break(subset_identifier)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()