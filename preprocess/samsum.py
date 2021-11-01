import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class SAMSum(FewshotGymTextToTextDataset):

    # pip install py7zr
    
    def __init__(self):
        self.hf_identifier = "samsum"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("summarize: " + datapoint["dialogue"].replace("\r\n", " ").replace("\n", " "), datapoint["summary"].replace("\r\n", " ").replace("\n", " ")))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("samsum")

def main():
    dataset = SAMSum()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()