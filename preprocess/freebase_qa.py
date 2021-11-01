import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class FreebaseQA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "freebase_qa"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if "RawQuestion" not in datapoint or "Parses" not in datapoint:
                continue
            input_text = datapoint["RawQuestion"]

            all_answers = []
            for item in datapoint["Parses"]["Answers"]: # why the file looks so weird...
                for answer_name in item["AnswersName"]:
                    for what in answer_name:
                        all_answers.append(what)
            all_answers = sorted(list(set(all_answers)))
            
            output_text = datapoint["Parses"]["Answers"][0]["AnswersName"][0][0]
            lines.append((input_text, "\t".join(all_answers)))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('freebase_qa')

def main():
    dataset = FreebaseQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()