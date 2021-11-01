# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
import re
import string
import argparse

from tqdm import tqdm

from utils import normalize_answer, download_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_test', default=False, action='store_true'),

    parser.add_argument('--max_length', default=200, type=int)
    parser.add_argument('--train_k', default=16384, type=int)
    parser.add_argument('--test_k', default=16, type=int)

    args = parser.parse_args()
    return args

prefix = "unifiedqa:"

def main():
    args = parse_args()

    if not os.path.exists(os.path.join(args.data_dir, "unifiedqa.zip")):
        download_file("1fybWU3eeN1jwFPuNa13JrLHzl5NGcYfB", os.path.join(args.data_dir, "unifiedqa.zip"))

    with open("../config/qa_to_qa.json", "r") as f:
        config = json.load(f)

    if args.do_train:
        train_tasks = [t[len(prefix):] for t in config["train"] if t.startswith(prefix)]
        process_train(args.data_dir, train_tasks, args.max_length, k=args.train_k, seed=100)

    if args.do_test:
        test_tasks = [t[len(prefix):] for t in config["test"] if t.startswith(prefix)]
        process_test(args.data_dir, test_tasks, k=args.test_k, seeds=[100, 13, 21, 42, 87])

def process_train(data_dir, train_tasks, max_length, k, seed):
    for task in train_tasks:
        np.random.seed(seed)

        data, lines = [], []
        with open(os.path.join(data_dir, prefix+task, "train.tsv"), "r") as f:
            for line in f:
                lines.append(line)
        for line in tqdm(lines):
            try:
                input_, output_ = line.strip().split("\t")
            except Exception:
                continue
            if task in ["natural_questions_with_dpr_para", "race_string", "drop", "newsqa",
                        "narrativeqa", "quoref", "ropes"]:
                if normalize_answer(output_) not in normalize_answer(input_):
                    continue

            if task in ["race_string", "social_iqa"]:
                in1, in2, in3 = input_.split("\\n")
                input_ = in1 + "\\n" + in3
            else:
                assert input_.count("\\n")<3
            if task in ["natural_questions_with_dpr_para"]:
                input_ = input_.replace(" , ", ", ").replace("( ", "(").replace(" )", ")").replace(" - - ", " - ").replace(" . ", ". ")

            if task in ["natural_questions_with_dpr_para", "race_string", "drop", "newsqa",
                        "narrativeqa", "quoref", "ropes"] and len(input_.split(" "))>max_length:
                question, context = input_.split("\\n")
                if normalize_answer(output_) not in normalize_answer(context):
                    #print (task)
                    #print (question)
                    #print (output_)
                    #print (context[:100])
                    continue
                n_words_question = len(question.split(" "))
                n_words_context = len(context.split(" "))
                n_words = max_length - n_words_question - 1
                assert n_words_context > n_words
                n_tries = 0
                while True:
                    start = np.random.choice(range(n_words_context-n_words+1))
                    new_context = " ".join(context.split(" ")[start:start+n_words])
                    if normalize_answer(output_) in normalize_answer(new_context):
                        input_ = question + " \\t " + new_context
                        break
                    n_tries += 1
                    #if n_tries % 1000 == 0:
                    #    print (n_tries, start, n_words_context, n_words)

            if len(output_.split(" "))>100:
                continue

            data.append(input_+"\t"+output_)

        data = [data[i] for i in np.random.permutation(range(len(data)))[:k]]
        lengths = []
        with open(os.path.join(data_dir, prefix+task, "{}_{}_{}_train.jsonl".format(prefix+task, k, seed)), "w") as f:
            for line in data:
                input_, output = line.split("\t")
                f.write(json.dumps({"task": prefix+task, "options": [], "input": input_, "output": output})+"\n")
                lengths.append(len(input_.split(" ")))
        print ("Finish saving %s\t#=%d" % (task, len(data)))

def process_test(data_dir, test_tasks, k, seeds):
    def _get_sentences(input_, options):
        sentences = []
        text = input_
        for option in options:
            if option not in text:
                break
            text1, text = text.split(option, 1)
            sentences.append(text1)
        sentences.append(text)
        return [s.strip() for s in sentences if len(s.strip())>0]

    def _prepro(task, split, line):
        line = line.strip()
        try:
            input_, output = line.split("\t")
        except Exception:
            print (line)
            exit()
        if line.count("\\n")==2:
            input_, options, context = input_.split("\\n")
            input_ = input_ + "\\n  " + context + " \\n " + options

        if input_.split("\\n")[-1].strip().startswith("(A)"):
            alphabet_options = list(string.ascii_uppercase)
            alphabet_options = ["(" + option + ")" for option in alphabet_options]
            option_text = input_.split("\\n")[-1].strip()
            options = _get_sentences(option_text, alphabet_options)
            assert output in options
            input_ = " ".join(input_.split("\\n")[:-1])
        elif output in ["yes", "no"]:
            options = []
            pass
        else:
            raise NotImplementedError()
        return json.dumps({"task": prefix+task, "input": input_, "options": options, "output": output})

    for task in test_tasks:
        with open(os.path.join(data_dir, prefix+task, "train.tsv"), "r") as f:
            data = [_prepro(task, "train", line) for line in f if len(line.strip())>0]
        with open(os.path.join(data_dir, prefix+task, "dev.tsv" if task in ["mctest", "multirc", "qasc", "qasc_with_ir"] else "test.tsv"), "r") as f:
            test_data = [_prepro(task, "test", line) for line in f if len(line.strip())>0]

        n_lengths = []
        for dp in data:
            n_lengths.append(len(dp.split(" ")))
        #print (task, len(n_lengths), "%.1f %.1f" % (np.mean(n_lengths), np.quantile(n_lengths, 0.90)))

        for seed in seeds:
            np.random.seed(seed)
            train_data = [data[i] for i in np.random.permutation(range(len(data)))[:k]]
            with open(os.path.join(data_dir, prefix+task, "{}_{}_{}_train.jsonl".format(prefix+task, k, seed)), "w") as f:
                for line in train_data:
                    f.write(line+"\n")
            with open(os.path.join(data_dir, prefix+task, "{}_{}_{}_test.jsonl".format(prefix+task, k, seed)), "w") as f:
                for line in test_data:
                    f.write(line+"\n")
        print ("Finish saving %s\t#=%d" % (task, k))

if __name__=='__main__':
    main()
