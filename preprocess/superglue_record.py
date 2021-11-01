import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Superglue_Record(FewshotGymTextToTextDataset):
    
    def __init__(self):
        self.hf_identifier = "superglue-record"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("question: " + datapoint["query"] + " context: " + datapoint["passage"].replace("\n", " "), datapoint["answers"][0]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("super_glue", "record")

def main():
    dataset = Superglue_Record()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()