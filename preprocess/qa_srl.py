# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class QA_SRL(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "qa_srl"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("question: " + " ".join(datapoint["question"]) + " context: " + datapoint["sentence"], "\t".join(datapoint["answers"])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('qa_srl')

def main():
    dataset = QA_SRL()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()