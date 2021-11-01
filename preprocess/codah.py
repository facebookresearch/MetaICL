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

class CODAH(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "codah"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["correct_answer_idx"]
        choices_string = ""
        for idx, candidate in enumerate(datapoint["candidate_answers"]):
            if idx == answer_index:
                answer_string = candidate
            choices_string += " " + id2alphabet[idx] + " " + candidate
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            lines.append((datapoint["question_propmt"] + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("codah", "fold_0")

def main():
    dataset = CODAH()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()