# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class LAMA(FewshotGymTextToTextDataset):

    def __init__(self, subset_name):
        self.hf_identifier = "lama-" + subset_name
        self.subset_name = subset_name
        self.task_type = "text to text"
        self.license = "unknown"

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)
        
        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        if self.subset_name == "trex": # trex is too large
            test_lines = lines[int(0.99*n):]
        else:
            test_lines = lines[int(0.8*n):]
        

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if self.subset_name in ["trex", "google_re"]:
                input_text = datapoint["template"].replace("[X]", datapoint["sub_label"]).replace("[Y]", "[MASK]")
            else:
                input_text = datapoint["masked_sentence"]
            lines.append((input_text, datapoint["obj_label"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("lama", self.subset_name)

def main():
    for subset in ["trex", "squad", "google_re", "conceptnet"]: 
        dataset = LAMA(subset)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()