# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class ClimateFever(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "climate_fever"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "Supports",
            1: "Refutes",
            2: "Not enough info",
            3: "Disputed",
        }

    def get_train_test_lines(self, dataset):
        # for some reason it only has a test set?
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        lines = map_hf_dataset_to_list(dataset, "test")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["claim"], self.label[datapoint["claim_label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('climate_fever')

def main():
    dataset = ClimateFever()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
