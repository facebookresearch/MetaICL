import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class BoolQ(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "boolq"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "no",
            1: "yes",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("question: " + datapoint["question"] + " [SEP] context: " + datapoint["passage"], self.label[datapoint["answer"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('boolq')

def main():
    dataset = BoolQ()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()