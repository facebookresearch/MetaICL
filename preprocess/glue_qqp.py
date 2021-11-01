import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Glue_QQP(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "glue-qqp"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "not_duplicate",
            1: "duplicate",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("question 1: " + datapoint["question1"] + " [SEP] question 2: " + datapoint["question2"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('glue', 'qqp')

def main():
    dataset = Glue_QQP()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

def main_more_shots():
    dataset = Glue_QQP()

    for shots in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=shots, seed=seed, path="../data_more_shots/{}_shot".format(str(shots)))


if __name__ == "__main__":
    main()
    # main_more_shots()