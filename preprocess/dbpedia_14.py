import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class DBpedia14(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "dbpedia_14"
        self.task_type = "classification"
        self.license = "cc-by-sa-3.0"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"Company",
            1:"EducationalInstitution",
            2:"Artist",
            3:"Athlete",
            4:"OfficeHolder",
            5:"MeanOfTransportation",
            6:"Building",
            7:"NaturalPlace",
            8:"Village",
            9:"Animal",
            10:"Plant",
            11:"Album",
            12:"Film",
            13:"WrittenWork",
        }

    def get_train_test_lines(self, dataset):
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list

        train_lines = map_hf_dataset_to_list(dataset, "train")
        test_lines = map_hf_dataset_to_list(dataset, "test")

        np.random.seed(42)
        np.random.shuffle(test_lines)
        n = len(test_lines)
        test_lines = test_lines[:int(0.05*n)]
        # using 10% of test cases, otherwise it's too slow to do evaluation

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["content"].replace("\t", " ").strip(), self.label[datapoint["label"]]))
            '''lines.append(json.dumps({
                "input": datapoint["content"].strip(),
                "output": self.label[datapoint["label"]],
                "choices": list(self.label.values())}))'''
        return lines

    def load_dataset(self):
        return datasets.load_dataset('dbpedia_14')

def main():
    dataset = DBpedia14()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
