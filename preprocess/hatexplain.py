# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset
from utils import get_majority

class HatExplain(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "hatexplain"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"hatespeech",
            1:"normal",
            2:"offensive",
        }

        self.license = "cc-by-4.0"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            label = get_majority(datapoint["annotators"]["label"])
            if label is not None:
                lines.append((" ".join(datapoint["post_tokens"]), self.label[label]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('hatexplain')

def main():
    dataset = HatExplain()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
