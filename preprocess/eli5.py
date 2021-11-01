import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class ELI5(FewshotGymTextToTextDataset):

    def __init__(self, subreddit):
        self.hf_identifier = "eli5-" + subreddit
        self.subreddit = subreddit
        self.task_type = "text to text"
        self.license = "unknown"

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train_{}".format(self.subreddit))
        test_lines = self.map_hf_dataset_to_list(dataset, "validation_{}".format(self.subreddit))

        np.random.seed(42)
        np.random.shuffle(test_lines)
        n = len(test_lines)
        test_lines = test_lines[:int(0.1*n)]
        # using 10% of test cases, otherwise it's too slow to do evaluation
        
        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = datapoint["title"].replace("\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ") + " [SEP] " + datapoint["selftext"].replace("\n", " ").replace("\r", " ").replace("\t", " ")
            lines.append((input_text, "\t".join(item.replace("\n", " ").replace("\r", " ").replace("\t", " ") for item in datapoint["answers"]["text"])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('eli5')

def main():
    for subreddit in ["eli5", "asks", "askh"]:
        dataset = ELI5(subreddit)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()