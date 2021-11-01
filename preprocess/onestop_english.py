# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class OneStopEnglish(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "onestop_english"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"elementary",
            1:"intermediate",
            2:"advance",
        }
    
    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

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
            datapoint["text"] = datapoint["text"].replace("\n", " ")
            if datapoint["text"].startswith("Intermediate  "): # some bug?
                datapoint["text"] = datapoint["text"][14:]
            lines.append((datapoint["text"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('onestop_english')

def main():
    dataset = OneStopEnglish()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()