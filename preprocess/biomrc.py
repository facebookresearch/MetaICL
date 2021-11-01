# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class BioMRC(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "biomrc"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_train_test_lines(self, dataset):

        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")

        np.random.seed(42)
        np.random.shuffle(test_lines)
        n = len(test_lines)
        test_lines = test_lines[:int(0.1*n)]
        # using 10% of test cases, otherwise it's too slow to do evaluation

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("question: " + datapoint["title"].replace("\n", " ") + " context: " + datapoint["abstract"].replace("\n", " "), datapoint["answer"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('biomrc', "biomrc_large_B")

def main():
    dataset = BioMRC()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()