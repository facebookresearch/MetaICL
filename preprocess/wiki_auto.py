import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class WikiAuto(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "wiki_auto"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"notAligned",
            1:"aligned",
        }

    def get_train_test_lines(self, dataset):

        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "dev")

        return train_lines, test_lines
        
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("normal sentence: " + datapoint["normal_sentence"] + " [SEP] simple_sentence: " + datapoint["simple_sentence"], self.label[datapoint["alignment_label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('wiki_auto', 'manual')

def main():
    dataset = WikiAuto()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()