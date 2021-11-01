# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

LABELS = [
    "[no-conn]",
    "absolutely,",
    "accordingly",
    "actually,",
    "additionally",
    "admittedly,",
    "afterward",
    "again,",
    "already,",
    "also,",
    "alternately,",
    "alternatively",
    "although,",
    "altogether,",
    "amazingly,",
    "and",
    "anyway,",
    "apparently,",
    "arguably,",
    "as_a_result,",
    "basically,",
    "because_of_that",
    "because_of_this",
    "besides,",
    "but",
    "by_comparison,",
    "by_contrast,",
    "by_doing_this,",
    "by_then",
    "certainly,",
    "clearly,",
    "coincidentally,",
    "collectively,",
    "consequently",
    "conversely",
    "curiously,",
    "currently,",
    "elsewhere,",
    "especially,",
    "essentially,",
    "eventually,",
    "evidently,",
    "finally,",
    "first,",
    "firstly,",
    "for_example",
    "for_instance",
    "fortunately,",
    "frankly,",
    "frequently,",
    "further,",
    "furthermore",
    "generally,",
    "gradually,",
    "happily,",
    "hence,",
    "here,",
    "historically,",
    "honestly,",
    "hopefully,",
    "however",
    "ideally,",
    "immediately,",
    "importantly,",
    "in_contrast,",
    "in_fact,",
    "in_other_words",
    "in_particular,",
    "in_short,",
    "in_sum,",
    "in_the_end,",
    "in_the_meantime,",
    "in_turn,",
    "incidentally,",
    "increasingly,",
    "indeed,",
    "inevitably,",
    "initially,",
    "instead,",
    "interestingly,",
    "ironically,",
    "lastly,",
    "lately,",
    "later,",
    "likewise,",
    "locally,",
    "luckily,",
    "maybe,",
    "meaning,",
    "meantime,",
    "meanwhile,",
    "moreover",
    "mostly,",
    "namely,",
    "nationally,",
    "naturally,",
    "nevertheless",
    "next,",
    "nonetheless",
    "normally,",
    "notably,",
    "now,",
    "obviously,",
    "occasionally,",
    "oddly,",
    "often,",
    "on_the_contrary,",
    "on_the_other_hand",
    "once,",
    "only,",
    "optionally,",
    "or,",
    "originally,",
    "otherwise,",
    "overall,",
    "particularly,",
    "perhaps,",
    "personally,",
    "plus,",
    "preferably,",
    "presently,",
    "presumably,",
    "previously,",
    "probably,",
    "rather,",
    "realistically,",
    "really,",
    "recently,",
    "regardless,",
    "remarkably,",
    "sadly,",
    "second,",
    "secondly,",
    "separately,",
    "seriously,",
    "significantly,",
    "similarly,",
    "simultaneously",
    "slowly,",
    "so,",
    "sometimes,",
    "soon,",
    "specifically,",
    "still,",
    "strangely,",
    "subsequently,",
    "suddenly,",
    "supposedly,",
    "surely,",
    "surprisingly,",
    "technically,",
    "thankfully,",
    "then,",
    "theoretically,",
    "thereafter,",
    "thereby,",
    "therefore",
    "third,",
    "thirdly,",
    "this,",
    "though,",
    "thus,",
    "together,",
    "traditionally,",
    "truly,",
    "truthfully,",
    "typically,",
    "ultimately,",
    "undoubtedly,",
    "unfortunately,",
    "unsurprisingly,",
    "usually,",
    "well,",
    "yet,",
]

class Discovery(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "discovery"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = LABELS

    def get_train_test_lines(self, dataset):
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list

        train_lines = map_hf_dataset_to_list(dataset, "train")
        test_lines = map_hf_dataset_to_list(dataset, "validation")

        np.random.seed(42)
        np.random.shuffle(test_lines)
        n = len(test_lines)
        test_lines = test_lines[:int(0.1*n)]
        # using 10% of test cases, otherwise it's too slow to do evaluation

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("sentence 1: " + datapoint["sentence1"] + " [SEP] sentence 2: " + datapoint["sentence2"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('discovery', 'discovery')

def main():
    dataset = Discovery()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
