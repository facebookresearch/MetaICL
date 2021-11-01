import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

id2alphabet = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}

class HellaSwag(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "hellaswag"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_index = int(datapoint["label"])
        choices_string = ""
        for i in range(len(datapoint["endings"])):
            if i == answer_index:
                answer_string = datapoint["endings"][i]
            choices_string += " " + id2alphabet[i] + " " + datapoint["endings"][i]
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            lines.append((datapoint["ctx"] + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("hellaswag")

def main():
    dataset = HellaSwag()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()