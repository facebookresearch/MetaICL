# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Ethos(FewshotGymClassificationDataset):
    def __init__(self, dimension):
        self.hf_identifier = "ethos-" + dimension
        self.dimension = dimension
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        if dimension == "violence":
            self.label = {
                0: "not violent",
                1: "violent",
            }
        elif dimension == "directed_vs_generalized":
            self.label = {
                0:"generalied",
            1:"directed",
            }
        else:
            self.label = {
                0:"false",
                1:"true",
            }

    def get_train_test_lines(self, dataset):
        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["text"].strip(), self.label[datapoint[self.dimension]]))
            #lines.append(json.dumps({
            #    "input": datapoint["text"].strip(),
            #    "output": self.label[datapoint[self.dimension]],
            #    "options": list(self.label.values()}))


        return lines

    def load_dataset(self):
        return datasets.load_dataset('ethos', 'multilabel')

def main():
    for dimension in ["directed_vs_generalized", "disability", "gender", "national_origin", "race", "religion", "sexual_orientation"]:
        dataset = Ethos(dimension)
        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
