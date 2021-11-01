# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class QUAREL(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "quarel"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_answer_string(self, datapoint):
        answer_index = datapoint["answer_index"]
        st1 = datapoint["question"].find("(A)")
        st2 = datapoint["question"].find("(B)")

        if answer_index == 0:
            answer_string = datapoint["question"][st1+4: st2]
        else:
            answer_string = datapoint["question"][st2+4: ]

        if answer_string.endswith("or "):
            answer_string = answer_string[:-3]

        return answer_string
        

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            answer_string = self.get_answer_string(datapoint)
            lines.append((datapoint["question"], answer_string.strip()))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("quarel")

def main():
    dataset = QUAREL()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()