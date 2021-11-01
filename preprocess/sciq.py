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

class SciQ(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "sciq"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_string = datapoint["correct_answer"]
        all_answers = [datapoint["distractor1"], datapoint["distractor2"], datapoint["distractor3"], answer_string]
        np.random.shuffle(all_answers)

        choices_string = ""
        for i, ans in enumerate(all_answers):
            choices_string += " " + id2alphabet[i] + " " + ans
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        np.random.seed(42)

        lines = []
        for datapoint in hf_dataset[split_name]:
            if len(datapoint["support"].replace("\n", " ")) == 0:
                continue
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            input_text = datapoint["question"].replace("\n", " ") + choices_string + " [SEP] " + datapoint["support"].replace("\n", " ") 
            lines.append((input_text, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("sciq")

def main():
    dataset = SciQ()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()