# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Quartz(FewshotGymTextToTextDataset):

    def __init__(self, mode):
        self.hf_identifier = "quartz-" + mode
        self.mode = mode
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["answerKey"]
        choices_string = ""
        for i in range(len(datapoint["choices"]["label"])):
            if datapoint["choices"]["label"][i] == answer_index:
                answer_string = datapoint["choices"]["text"][i]
            choices_string += " (" + datapoint["choices"]["label"][i] + ") " + datapoint["choices"]["text"][i]
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            if self.mode == "with_knowledge":
                input_text = datapoint["question"] + datapoint["para"] + choices_string
            elif self.mode == "no_knowledge":
                input_text = datapoint["question"] + choices_string
            else:
                raise Exception()
            lines.append((input_text, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("quartz")

def main():
    for mode in ["with_knowledge", "no_knowledge"]:
        dataset = Quartz(mode)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()