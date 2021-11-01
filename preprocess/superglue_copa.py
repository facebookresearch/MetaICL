# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Superglue_COPA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "superglue-copa"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["label"]
        choices_string = " (A) " + datapoint["choice1"] + " (B) " + datapoint["choice2"]
        if answer_index == 0:
            answer_string = datapoint["choice1"]
        else:
            answer_string = datapoint["choice2"]
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            lines.append((datapoint["premise"] + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("super_glue", "copa")

def main():
    dataset = Superglue_COPA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()