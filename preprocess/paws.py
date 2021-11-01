# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class PAWS(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "paws"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "not_duplicate",
            1: "duplicate",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("sentence 1: " + datapoint["sentence1"] + " [SEP] sentence 2: " + datapoint["sentence2"],
                          self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('paws', 'labeled_final')

def main():
    dataset = PAWS()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
