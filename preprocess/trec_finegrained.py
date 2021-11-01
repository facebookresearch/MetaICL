import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class TREC_Finegrained(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "trec-finegrained"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"manner",
            1:"cremat",
            2:"animal",
            3:"exp",
            4:"ind",
            5:"gr",
            6:"title",
            7:"def",
            8:"date",
            9:"reason",
            10:"event",
            11:"state",
            12:"desc",
            13:"count",
            14:"other",
            15:"letter",
            16:"religion",
            17:"food",
            18:"country",
            19:"color",
            20:"termeq",
            21:"city",
            22:"body",
            23:"dismed",
            24:"mount",
            25:"money",
            26:"product",
            27:"period",
            28:"substance",
            29:"sport",
            30:"plant",
            31:"techmeth",
            32:"volsize",
            33:"instru",
            34:"abb",
            35:"speed",
            36:"word",
            37:"lang",
            38:"perc",
            39:"code",
            40:"dist",
            41:"temp",
            42:"symbol",
            43:"ord",
            44:"veh",
            45:"weight",
            46:"currency",
        }

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
            lines.append((datapoint["text"].replace("\t", "").replace("\n", "").replace("\r", ""), self.label[datapoint["label-fine"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('trec')

def main():
    dataset = TREC_Finegrained()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
