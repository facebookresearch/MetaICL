import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class TweetEval(FewshotGymClassificationDataset):
    def __init__(self, subset_name):
        self.hf_identifier = "tweet_eval-" + subset_name
        self.subset_name = subset_name
        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        if subset_name == "emoji":
            self.label = {
                0:"❤",
                1:"😍",
                2:"😂",
                3:"💕",
                4:"🔥",
                5:"😊",
                6:"😎",
                7:"✨",
                8:"💙",
                9:"😘",
                10:"📷",
                11:"🇺🇸",
                12:"☀",
                13:"💜",
                14:"😉",
                15:"💯",
                16:"😁",
                17:"🎄",
                18:"📸",
                19:"😜",
            }
        elif subset_name == "emotion":
            self.label = {
                0:"anger",
                1:"joy",
                2:"optimism",
                3:"sadness",
            }
        elif subset_name == "hate":
            self.label = {
                0:"non-hate",
                1:"hate",
            }
        elif subset_name == "irony":
            self.label = {
                0:"non-irony",
                1:"hate",
            }
        elif subset_name == "offensive":
            self.label = {
                0:"non-offensive",
                1:"hate",
            }
        elif subset_name == "sentiment":
            self.label = {
                0:"negative",
                1:"neutral",
                2:"positive",
            }
        else:
            self.label = {
                0:"none",
                1:"against",
                2:"favor",
            }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            if len(datapoint["text"].replace("\n", " ")):
                lines.append((datapoint["text"].replace("\n", " "), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('tweet_eval', self.subset_name)

def main():
    for subset_name in ['emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment', 'stance_abortion', 'stance_atheism', 'stance_climate', 'stance_feminist', 'stance_hillary']:
        dataset = TweetEval(subset_name)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
