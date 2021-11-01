import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

ID2ALPHABET = {i : "(" + chr(65+i) + ")" for i in range(26)}

class Superglue_MultiRC(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "superglue-multirc"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_source_and_target_string(self, paragraph):
        src = "question: {}".format(paragraph[0].replace("\n", " ").replace("\r", " ").replace("\t", " "))

        for idx, choice in enumerate(paragraph[2]):
            src += " " + ID2ALPHABET[idx] + " " + choice.replace("\n", " ").replace("\r", " ").replace("\t", " ")

        src += " [SEP] context: {}".format(paragraph[1].replace("\n", " ").replace("\r", " ").replace("\t", " "))

        correct_answers = []
        for answer, label in zip(paragraph[2], paragraph[3]):
            if label == 1:
                correct_answers.append(answer)
        if len(correct_answers) == 0:
            tar = "NO ANSWER!"
        else:
            tar = "\t".join(correct_answers)

        return src, tar

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []

        paragraphs = {}
        for datapoint in hf_dataset[split_name]:
            if datapoint["idx"]["question"] not in paragraphs:
                paragraphs[datapoint["idx"]["question"]] = [datapoint["question"], datapoint["paragraph"], [datapoint["answer"]], [datapoint["label"]]]
            else:
                paragraphs[datapoint["idx"]["question"]][2].append(datapoint["answer"])
                paragraphs[datapoint["idx"]["question"]][3].append(datapoint["label"])

        for paragraph in paragraphs.values():
            lines.append(self.get_source_and_target_string(paragraph))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("super_glue", "multirc")

def main():
    dataset = Superglue_MultiRC()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
