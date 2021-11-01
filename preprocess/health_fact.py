# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class HealthFact(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "health_fact"
        self.task_type = "classification"
        self.license = "mit"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"false",
            1:"mixture",
            2:"true",
            3:"unproven",
        }
    
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if datapoint["label"] < 0:
                continue
            lines.append((datapoint["claim"].strip().replace("\n", " ").replace("\r", " ").replace("\t", " "), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('health_fact')

def main():
    dataset = HealthFact()
    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()