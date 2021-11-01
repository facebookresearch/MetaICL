import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class SQuAD(FewshotGymTextToTextDataset):

    def __init__(self, mode):
        self.hf_identifier = "squad-" + mode
        self.mode = mode
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if self.mode == "with_context":
                lines.append(("question: " + datapoint["question"] + " context: " + datapoint["context"].replace("\t", " ").replace("\n", " "), datapoint["answers"]["text"][0]))
            else:
                lines.append(("question: " + datapoint["question"], datapoint["answers"]["text"][0]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("squad")

def main():
    for mode in ["with_context", "no_context"]:
        dataset = SQuAD(mode)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
