import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Superglue_Wic(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "superglue-wic"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "false",
            1: "true",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("word: " + datapoint["word"] + " [SEP] sentence 1: " + datapoint["sentence1"] + " [SEP] sentence 2: " + datapoint["sentence2"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('super_glue', 'wic')

def main():
    dataset = Superglue_Wic()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()