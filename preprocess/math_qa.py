# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset


class MathQA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "math_qa"
        self.task_type = "text to text"
        self.license = "unknown"

    def process_line(self, dp):
        options = dp["options"].split(",")
        choices = " (A) " + options[0][4:-1]
        if dp["correct"] == "a":
            answer = options[0][4:-1]
        choices += " (B) " + options[1][5:-1]
        if dp["correct"] == "b":
            answer = options[1][5:-1]
        choices += " (C) " + options[2][5:-1]
        if dp["correct"] == "c":
            answer = options[2][5:-1]
        choices += " (D) " + options[3][5:-1]
        if dp["correct"] == "d":
            answer = options[3][5:-1]
        choices += " (E) " + options[4][5:]
        if dp["correct"] == "e":
            answer = options[4][5:]
        
        return choices, answer
        

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices, answer = self.process_line(datapoint)
            if answer:
                lines.append((datapoint["Problem"] + choices, answer))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('math_qa')

def main():
    dataset = MathQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()