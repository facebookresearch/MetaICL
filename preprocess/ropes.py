# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class ROPES(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "ropes"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = "question: " + datapoint["question"] + " [SEP] situation: " + datapoint["situation"] + " [SEP] background: " + datapoint["background"]
            output_text = "\t".join(datapoint["answers"]["text"])
            lines.append((input_text.replace("\n", " "), output_text))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("ropes")

def main():
    dataset = ROPES()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

def main_more_shots():
    dataset = ROPES()

    for shots in [64, 128, 256, 512, 1024, 2048, 4096]:
        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=shots, seed=seed, path="../data_more_shots/{}_shot".format(str(shots)))

if __name__ == "__main__":
    main()
    # main_more_shots()