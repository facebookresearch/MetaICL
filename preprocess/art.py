# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class ART(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "art"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            1: "hypothesis 1",
            2: "hypothesis 2",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            input_line = "observation 1: " + datapoint["observation_1"] + " [SEP] observation 2: " + datapoint["observation_2"] + " [SEP] hypothesis 1: " + datapoint["hypothesis_1"] + " [SEP] hypothesis 2: " + datapoint["hypothesis_2"]
            lines.append((input_line.replace("\n", " ").replace("\t", " ") , self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('art')

def main():
    dataset = ART()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
