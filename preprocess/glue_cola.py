# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Glue_Cola(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "glue-cola"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "unacceptable",
            1: "acceptable",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["sentence"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('glue', 'cola')

def main():
    dataset = Glue_Cola()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()