# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Limit(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "limit"
        self.task_type = "sequence tagging" # ?
        self.license = "unknown"

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "test")

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if datapoint["motion"] != "yes":
                continue
            entities = [item["entity"] for item in datapoint["motion_entities"]]
            lines.append((datapoint["sentence"].strip(), " [SEP] ".join(entities)))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('limit')

def main():
    dataset = Limit()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()