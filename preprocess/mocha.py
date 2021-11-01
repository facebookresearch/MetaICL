# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Mocha(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "mocha"
        self.task_type = "regression"
        self.license = "unknown"


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if datapoint["score"] % 1 != 0:
                continue
            # line[0]: input; line[1]: output
            input_text = "question: " + datapoint["question"] + " [SEP] context: " + datapoint["context"] + " [SEP] reference: " + datapoint["reference"] + " [SEP] candidate" + datapoint["candidate"]
            lines.append((input_text, int(datapoint["score"])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('mocha')

def main():
    dataset = Mocha()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()