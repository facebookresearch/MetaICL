# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class WikiSplit(FewshotGymTextToTextDataset):
    
    def __init__(self):
        self.hf_identifier = "wiki_split"
        self.task_type = "text to text"
        self.license = "unknown"


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("sentence 1: " + datapoint["simple_sentence_1"] + " [SEP] sentence 2: " + datapoint["simple_sentence_2"], datapoint["complex_sentence"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("wiki_split")

def main():
    dataset = WikiSplit()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()