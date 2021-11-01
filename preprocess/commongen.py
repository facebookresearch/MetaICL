import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class CommonGen(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "common_gen"
        self.task_type = "text to text"
        self.license = "apache-2.0"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        d = {}
        for datapoint in hf_dataset[split_name]:
            if datapoint["concept_set_idx"] not in d:
                d[datapoint["concept_set_idx"]] = (datapoint["concepts"], [datapoint["target"]])
            else:
                d[datapoint["concept_set_idx"]][1].append(datapoint["target"])
        for k, v in d.items():
            lines.append((", ".join(v[0]), "\t".join(v[1])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('common_gen')

def main():
    dataset = CommonGen()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()