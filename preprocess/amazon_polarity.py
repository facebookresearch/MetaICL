# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import json
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class AmazonPolarity(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "amazon_polarity"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "negative",
            1: "positive",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list

        lines = map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:10000]
        test_lines = lines[10000:11000]

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        if split_name == "validation":
            split_name = "test" # hg datasets only has train/test
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("title: " + datapoint["title"] + " [SEP] content: " + datapoint["content"], self.label[datapoint["label"]]))
            #lines.append(json.dumps({
            #    "input": "title: " + datapoint["title"] + " content: " + datapoint["content"],
            #    "output": self.label[datapoint["label"]],
            #    "choices": list(self.labels.values())}))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('amazon_polarity')

def main():
    dataset = AmazonPolarity()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
