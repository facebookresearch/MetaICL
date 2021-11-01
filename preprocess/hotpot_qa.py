# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class HotpotQA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "hotpot_qa"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_context(self, dp):
        counter = 1
        context = ""
        titles = dp["supporting_facts"]["title"]
        for sentences, title in zip(dp["context"]["sentences"], dp["context"]["title"]):
            if title in titles:
                context += "".join(sentences) + " "
        return context

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            context = self.get_context(datapoint)
            lines.append(("question: " + datapoint["question"] + " context: " + context.strip(), datapoint["answer"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("hotpot_qa", "distractor")

def main():
    dataset = HotpotQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()