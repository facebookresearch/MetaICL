import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Circa(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "circa"
        self.task_type = "classification"
        self.license = "cc-by-4.0"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "Yes",
            1: "No",
            2: "In the middle, neither yes nor no",
            3: "Yes, subject to some conditions",
            4: "Other",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        lines = map_hf_dataset_to_list(dataset, "train")

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
            if datapoint["goldstandard2"] == -1:
                continue
            input_text = "context: " + datapoint["context"] + " [SEP] question X: " + datapoint["question-X"] + " [SEP] answer Y: " + datapoint["answer-Y"]
            lines.append((input_text, self.label[datapoint["goldstandard2"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('circa')

def main():
    dataset = Circa()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
