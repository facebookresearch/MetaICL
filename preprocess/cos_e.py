import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

id2alphabet = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}

class CoS_E(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "cos_e"
        self.task_type = "text to text"
        self.license = "BSD-3"

    def get_choices_and_answer_string(self, datapoint):
        answer_string = datapoint["answer"]
        choices_string = ""
        for idx, candidate in enumerate(datapoint["choices"]):
            choices_string += " " + id2alphabet[idx] + " " + candidate
        return choices_string, answer_string


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            lines.append((datapoint["question"].replace("\t", " ").replace("\n", " ") + choices_string,
                          datapoint["answer"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("cos_e", "v1.11")

def main():
    dataset = CoS_E()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
