# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
import argparse

from tqdm import tqdm
from collections import defaultdict

from utils import load_configs, load_prompts, apply_prompt, map_hf_dataset_to_list, preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--inst', action='store_true',
                    help="Construct data from hg datasets.")
parser.add_argument('--do_train', action='store_true',
                    help="Verify the datafiles with pre-computed MD5")
parser.add_argument('--do_test', action='store_true',
                    help="Run 2 tasks per process to test the code")

args = parser.parse_args()

use_instruct = args.inst
do_train = args.do_train
do_test = args.do_test
if args.do_train and args.do_test:
    raise NotImplementedError("You should specify one of `--do_train` and `--do_test`, not both")
if not args.do_train and not args.do_test:
    raise NotImplementedError("You should specify one of `--do_train` and `--do_test`")

config_dict = load_configs()
if use_instruct:
    prompt_names_per_task, prompt_dict = load_prompts(do_train)

class FewshotGymDataset():

    def get_map_hf_dataset_to_list(self):
        if use_instruct:
            def _map_hf_dataset_to_list(dataset, split):
                return map_hf_dataset_to_list(self.hf_identifier, dataset, split, do_train=do_train)
            return _map_hf_dataset_to_list
        return None

    def get_train_test_lines(self, dataset):
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        train_lines = map_hf_dataset_to_list(dataset, "train")
        test_lines = map_hf_dataset_to_list(dataset, "validation")
        return train_lines, test_lines

    def save(self, path, k, seed, k_shot_train, k_shot_dev, k_shot_test):
        # save to path

        def _apply_prompt(example):
            return apply_prompt(self.hf_identifier, example, do_train=do_train, prompt_names_per_task=prompt_names_per_task, prompt_dict=prompt_dict)

        if do_train and use_instruct:
            # let's save k_shot_train only

            grouped_k_shot_train = defaultdict(list)
            for line in tqdm(k_shot_train):
                line = _apply_prompt(line)
                assert type(line)==dict
                assert len(set(line.keys())-set(["inst:"+self.hf_identifier+":"+name for name in prompt_names_per_task[self.hf_identifier]]))==0

                for key, value in line.items():
                    grouped_k_shot_train[key].append(json.dumps(value))

            for key, lines in grouped_k_shot_train.items():
                hf_identifier = key
                if path:
                    os.makedirs(os.path.join(path, hf_identifier), exist_ok=True)
                    prefix = os.path.join(path, hf_identifier,
                                          "{}_{}_{}".format(hf_identifier, k, seed))
                    self.write(lines, prefix + "_train.jsonl")

        elif use_instruct:
            k_shot_train = [_apply_prompt(example) for example in k_shot_train]
            k_shot_dev = [_apply_prompt(example) for example in k_shot_dev]
            k_shot_test = [_apply_prompt(example) for example in k_shot_test]

            hf_identifier = "inst:"+self.hf_identifier if use_instruct else self.hf_identifier
            if path:
                os.makedirs(os.path.join(path, hf_identifier), exist_ok=True)
                prefix = os.path.join(path, hf_identifier,
                                    "{}_{}_{}".format(hf_identifier, k, seed))
                self.write(k_shot_train, prefix + "_train.jsonl")
                self.write(k_shot_dev, prefix + "_dev.jsonl")
                self.write(k_shot_test, prefix + "_test.jsonl")

        else:
            config = config_dict[self.hf_identifier]
            k_shot_train = [preprocess(self.hf_identifier, example, config) for example in k_shot_train]
            if do_test:
                k_shot_dev = [preprocess(self.hf_identifier, example, config) for example in k_shot_dev]
                k_shot_test = [preprocess(self.hf_identifier, example, config) for example in k_shot_test]

            if path:
                os.makedirs(os.path.join(path, self.hf_identifier), exist_ok=True)
                prefix = os.path.join(path, self.hf_identifier,
                                      "{}_{}_{}".format(self.hf_identifier, k, seed))
                self.write(k_shot_train, prefix + "_train.jsonl")
                self.write(k_shot_dev, prefix + "_dev.jsonl")
                self.write(k_shot_test, prefix + "_test.jsonl")

    def write(self, lst, out_file):
        with open(out_file, "w") as fout:
            for line in lst:
                if line is not None:
                    fout.write(line+"\n")

class FewshotGymClassificationDataset(FewshotGymDataset):

    def generate_k_shot_data(self, k, seed, path=None):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        if self.hf_identifier not in config_dict:
            return None, None, None

        if use_instruct and self.hf_identifier not in prompt_names_per_task:
            return None, None, None

        if do_train:
            if seed<100:
                return None, None, None
            k = 16384
        elif do_test:
            k = 16

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        # Get label list for balanced sampling
        label_list = {}
        for line in train_lines:
            label = "all"
            if label not in label_list:
                label_list[label] = [line]
            else:
                label_list[label].append(line)

        # make train, dev, test data
        k_shot_train = []
        for label in label_list:
            for line in label_list[label][:k]:
                k_shot_train.append(line)

        k_shot_dev = []
        for label in label_list:
            for line in label_list[label][k:2*k]:
                k_shot_dev.append(line)

        k_shot_test = test_lines

        # save to path
        self.save(path, k, seed, k_shot_train, k_shot_dev, k_shot_test)
        return k_shot_train, k_shot_dev, k_shot_test

class FewshotGymTextToTextDataset(FewshotGymDataset):

    def generate_k_shot_data(self, k, seed, path=None):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        if self.hf_identifier not in config_dict:
            return None, None, None

        if use_instruct and self.hf_identifier not in prompt_names_per_task:
            return None, None, None

        if do_train:
            if seed<100:
                return None, None, None
            k = 16384
        elif do_test:
            k = 16

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        # make train, dev, test data
        k_shot_train = []
        for line in train_lines[:k]:
            k_shot_train.append(line)

        k_shot_dev = []
        for line in train_lines[k:2*k]:
            k_shot_dev.append(line)

        k_shot_test = test_lines

        self.save(path, k, seed, k_shot_train, k_shot_dev, k_shot_test)
        return k_shot_train, k_shot_dev, k_shot_test
