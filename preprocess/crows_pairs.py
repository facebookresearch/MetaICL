import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class CrowsPairs(FewshotGymClassificationDataset):

    def __init__(self):
        self.hf_identifier = "crows_pairs"
        self.task_type = "classification"
        self.license = "unknown"

        self.label = {
            0: "sentence 1",
            1: "sentence 2"
        }

    def get_train_test_lines(self, dataset):
        # only test set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "test")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        np.random.seed(42)
        for datapoint in hf_dataset[split_name]:
            sent_more = datapoint["sent_more"].replace("\n", "").replace("\t", "")
            sent_less = datapoint["sent_less"].replace("\n", "").replace("\t", "")
            if np.random.random() > 0.5:
                lines.append(("sentence 1: " + sent_more + " [SEP] sentence 2: " + sent_less, self.label[0]))
            else:
                lines.append(("sentence 1: " + sent_less + " [SEP] sentence 2: " + sent_more, self.label[1]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('crows_pairs')

def main():
    dataset = CrowsPairs()
    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
