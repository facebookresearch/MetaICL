import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Emotion(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "emotion"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["text"].strip(), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('emotion')

def main():
    dataset = Emotion()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()