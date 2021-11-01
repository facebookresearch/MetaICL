import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class SciTail(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "scitail"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            if datapoint["gold_label"] == "entailment":
                label = 0
            elif datapoint["gold_label"] == "neutral":
                label = 1
            elif datapoint["gold_label"] == "contradiction":
                label = 2
            lines.append(("sentence 1: " + datapoint["sentence1"] + " [SEP] sentence 2: " + datapoint["sentence2"], datapoint["gold_label"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('scitail', 'snli_format')

def main():
    dataset = SciTail()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()