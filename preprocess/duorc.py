import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class DuoRC(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "duorc"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if datapoint["no_answer"] == 1:
                continue

            if datapoint["plot"].startswith("This article's plot summary may be too long or excessively detailed. Please help improve it by removing unnecessary details and making it more concise."):
                continue
                # datapoint["plot"] = datapoint["plot"].replace("This article's plot summary may be too long or excessively detailed. Please help improve it by removing unnecessary details and making it more concise.", "").strip(" ")
            # assert len(datapoint["answers"]) == 1
            input_text = "question: " + datapoint["question"] + " context: " + datapoint["plot"].replace("\n", " ")
            lines.append((input_text.replace("\n", " ").replace("\t", " "), "\t".join(datapoint["answers"])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('duorc', 'SelfRC')

def main():
    dataset = DuoRC()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
