# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

id2alphabet = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}

class Race(FewshotGymTextToTextDataset):

    def __init__(self, subset):
        self.hf_identifier = "race-" + subset
        self.subset = subset
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_index = ord(datapoint["answer"]) - ord("A")
        choices_string = ""
        for i, ans in enumerate(datapoint["options"]):
            if i == answer_index:
                answer_string = ans.replace("\n", " ").replace("\t", " ").replace("\r", " ")
            choices_string += " " + id2alphabet[i] + " " + ans.replace("\n", " ").replace("\t", " ").replace("\r", " ")
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            input_text = datapoint["question"].replace("\n", " ").replace("\t", " ").replace("\r", " ") + choices_string + " [SEP] " + datapoint["article"].replace("\n", " ").replace("\t", " ").replace("\r", " ")
            lines.append((input_text, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("race", self.subset)

def main():
    for subset in ["middle", "high"]:
        dataset = Race(subset)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()