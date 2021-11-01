import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Liar(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "liar"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"false",
            1:"half-true",
            2:"mostly-true",
            3:"true",
            4:"barely-true",
            5:"pants-fire",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            input_text = "statement: " + datapoint["statement"] + " [SEP] speaker: " + datapoint["speaker"] + " [SEP] context: " + datapoint["context"]
            lines.append((input_text.replace("\n", " ").replace("\r", " ").replace("\t", " "), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('liar')

def main():
    dataset = Liar()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()