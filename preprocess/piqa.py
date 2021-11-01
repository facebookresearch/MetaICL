# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class PIQA(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "piqa"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "solution 1",
            1: "solution 2"
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            input_ = "goal: " + datapoint["goal"] + " [SEP] solution 1" + datapoint["sol1"] + " [SEP] solution 2" + datapoint["sol2"]
            lines.append((input_.replace("\t", "").replace("\n", "").replace("\r", ""), datapoint["label"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('piqa')

def main():
    dataset = PIQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
