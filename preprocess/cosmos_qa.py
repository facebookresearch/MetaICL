# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

id2alphabet = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}

class CosmosQA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "cosmos_qa"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_key = "answer" + str(datapoint["label"])
        answer_string = datapoint[answer_key]
        choices_string = ""
        for idx in range(4):
            choices_string += " " + id2alphabet[idx] + " " + datapoint["answer" + str(idx)]
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            lines.append((datapoint["question"] + " [SEP] " + datapoint["context"] + " [SEP] " + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("cosmos_qa")

def main():
    dataset = CosmosQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()