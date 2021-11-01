# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class AquaRat(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "aqua_rat"
        self.task_type = "text to text"
        self.license = "apache 2.0"


    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["correct"]
        choices_string = ""
        for option in datapoint["options"]:
            if option[0] == answer_index:
                answer_string = option[2:]
            choices_string += " (" + option[0:2] + " " + option[2:]
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            lines.append((datapoint["question"].replace("\n", " ") + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("aqua_rat", "raw")

def main():
    dataset = AquaRat()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()