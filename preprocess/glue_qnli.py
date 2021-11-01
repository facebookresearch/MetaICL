import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Glue_QNLI(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "glue-qnli"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "entailment",
            1: "not_entailment",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("question: " + datapoint["question"] + " [SEP] sentence: " + datapoint["sentence"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('glue', 'qnli')

def main():
    dataset = Glue_QNLI()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()