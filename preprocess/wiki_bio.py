# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class WikiBio(FewshotGymTextToTextDataset):
    
    def __init__(self):
        self.hf_identifier = "wiki_bio"
        self.task_type = "text to text"
        self.license = "cc-by-3.0"

    def get_train_test_lines(self, dataset):

        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "val")

        np.random.seed(42)
        np.random.shuffle(test_lines)
        n = len(test_lines)
        test_lines = test_lines[:int(0.01*n)]
        # using 1% of test cases, otherwise it's too slow to do evaluation

        return train_lines, test_lines

    def make_input_text(self, datapoint):
        input_text = datapoint["input_text"]["context"].strip() + " [SEP] "
        if len(datapoint["input_text"]["table"]["column_header"]) != len(datapoint["input_text"]["table"]["content"]):
            return None
        for a, b in zip(datapoint["input_text"]["table"]["column_header"], datapoint["input_text"]["table"]["content"]):
            input_text += "{}: {} [n] ".format(a, b.strip().replace("\n", " "))
        return input_text

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = self.make_input_text(datapoint)
            if input_text:
                lines.append((input_text, datapoint["target_text"].replace("\n", " ")))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("wiki_bio")

def main():
    dataset = WikiBio()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()