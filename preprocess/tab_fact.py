import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class TabFact(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "tab_fact"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "refuted",
            1: "entailed",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("statement: " + datapoint["statement"] + " [SEP] table_caption: " + datapoint["table_caption"] + " [SEP] table_text: " + datapoint["table_text"].replace("\n", " [n] "), self.label[datapoint["label"]]))
            #lines.append(json.dumps({
            #    "input": "statement: " + datapoint["statement"] + " table_caption: " + datapoint["table_caption"] + " table_text: " + datapoint["table_text"].replace("\n", " [n] "),
            #    "output": self.label[datapoint["label"]],
            #    "options": list(self.label.values())}))

        return lines

    def load_dataset(self):
        return datasets.load_dataset('tab_fact', 'tab_fact')

def main():
    dataset = TabFact()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
