import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class AcronymIdentification(FewshotGymTextToTextDataset):

    # 0:"B-long", 1:"B-short", 2:"I-long", 3:"I-short", 4:"O"

    def __init__(self):
        self.hf_identifier = "acronym_identification"
        self.task_type = "sequence tagging"
        self.license = "CC BY-NC-SA 4.0"

        self.labels = {
            0:"B-long",
            1:"B-short",
            2:"I-long",
            3:"I-short",
            4:"O",
        }

    def get_acronym_and_full_name(self, dp):
        labels = dp["labels"]
        short_st = labels.index(1)
        short_ed = short_st
        while labels[short_ed + 1] == 3:
            short_ed += 1
        acronym = " ".join(dp["tokens"][short_st: short_ed+1])

        long_st = labels.index(0)
        long_ed = long_st
        while labels[long_ed + 1] == 2:
            long_ed += 1
        full_name = " ".join(dp["tokens"][long_st: long_ed+1])

        return acronym, full_name

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            assert len(datapoint["labels"]) == len(datapoint["tokens"])
            if datapoint["labels"].count(1) != 1 or datapoint["labels"].count(0) != 1: # only contains one acronym
                continue
            acronym, full_name = self.get_acronym_and_full_name(datapoint)
            #lines.append((" ".join(datapoint["tokens"]) + " [SEP] acronym: " + acronym, full_name))
            lines.append(json.dumps({
                "input": " ".join(datapoint["tokens"]) + " acronym: " + acronym,
                "output": full_name}))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('acronym_identification')

def main():
    dataset = AcronymIdentification()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
