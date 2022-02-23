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

    def get_choices_and_answer_string(self, datapoint, output_prefix):
        answer_index = datapoint["label"]
        choice1 = output_prefix + datapoint["choice1"]
        choice2 = output_prefix + datapoint["choice2"]
        choices_string = " (A) " + choice1 + " (B) " + choice2
        if answer_index == 0:
            answer_string = choice1
        else:
            answer_string = choice2
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            premise = datapoint["premise"].strip()
            if datapoint["question"]=="cause":
                input_prefix = "Effect: "
                output_prefix = "Cause: "
            elif datapoint["question"]=="effect":
                input_prefix = "Cause: "
                output_prefix = "Effect: "
            else:
                raise NotImplementedError()
            premise = input_prefix + premise
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint, output_prefix)

            lines.append((premise + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("super_glue", "copa")

def main():
    dataset = Superglue_COPA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
