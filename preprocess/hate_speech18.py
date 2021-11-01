import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class HateSpeech18(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "hate_speech18"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "noHate",
            1: "hate",
        }

        self.license = "cc-by-sa-3.0"

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            if datapoint["label"] > 1: # only deal with hate/nohate
                continue
            lines.append((datapoint["text"], self.label[datapoint["label"]]))
            #lines.append(json.dumps({
            #    "input": datapoint["text"],
            #    "output": self.label[datapoint["label"]],
            #    "options": list(self.label.values())}))

        return lines

    def load_dataset(self):
        return datasets.load_dataset('hate_speech18')

def main():
    dataset = HateSpeech18()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
