import os
import json
import string
import re
import subprocess

from collections import defaultdict, Counter

def get_majority(lst):
    c = Counter(lst)
    rank = c.most_common()
    if len(rank) == 1:
        return rank[0][0]
    elif rank[0][1] == rank[1][1]:
        return None
    else:
        return rank[0][0]

def load_configs():
    config_dict = {}
    for task in os.listdir("../config/tasks"):
        if not task.startswith("unifiedqa:"):
            with open(os.path.join("../config/tasks", task), "r") as f:
                config = json.load(f)
            config_dict[task.split(".")[0]] = config
    return config_dict

def load_prompts(do_train):
    from promptsource.templates import TemplateCollection

    subtask_dict = {
        "ai2_arc": "ARC-Challenge",
        "codah": "fold_0",
        "hotpot_qa": "disctractor",
        "openbookqa": "main",
        "paws": "labeled_final",
        "scitail": "snli_format"
    }
    with open("../config/hr_to_lr_noinst.json", "r") as f:
        config = json.load(f)
    train_tasks = set(config["train"])
    test_tasks = set(config["test"])

    if do_train:
        test_tasks = set()
    else:
        train_tasks = set()

    collection = TemplateCollection()
    available_tasks = defaultdict(list)
    prompt_names_per_task = {}
    prompt_dict = {}
    for task, subtask in collection.keys:
        if task in train_tasks or task in test_tasks:
            available_tasks[task].append(subtask)

    for task, subtasks in available_tasks.items():
        if len(subtasks)>1:
            subtasks = [subtask_dict[task]]
        assert len(subtasks)==1, (task, subtasks)
        available_tasks[task] = subtasks[0]

    def normalize_name(name):
        return name.replace(" ", "-").replace("/", "-").replace("_", "-")

    for task, subtask in available_tasks.items():
        prompts = collection.get_dataset(task, subtask)

        if do_train:
            prompt_names_per_task[task] = []
            for name in prompts.all_template_names:
                if task=="circa" and name in ["possible_qn", "question_declarative"]:
                    # always give empty output for some reason
                    print ("Skipping", task, name)
                    continue
                prompt_names_per_task[task].append(normalize_name(name))
                prompt_dict[task+":"+normalize_name(name)] = prompts[name]
        else:

            all_template_names = [name for name in prompts.all_template_names if "no_option" not in name]

            if task=="dream":
                all_template_names = ["read_the_following_conversation_and_answer_the_question"]

            for keyword in ["multiple_choice", "most_correct", "most_suitable"]:
                _all_template_names = [name for name in all_template_names if keyword in name]
                if len(_all_template_names)>0:
                    all_template_names = _all_template_names

            if len(all_template_names)<1:
                continue

            prompt = prompts[all_template_names[0]]
            prompt_names_per_task[task] = [all_template_names[0]]
            prompt_dict[task] = prompt

    with open("../config/hr_to_lr_inst_all.json", "r") as f:
        config = json.load(f)
    datasets = [t[5:] for t in config["train" if do_train else "test"]]

    assert set(datasets)==set(prompt_dict.keys()), (len(datasets), len(prompt_dict))

    return prompt_names_per_task, prompt_dict


def apply_prompt(task, example, do_train, prompt_names_per_task, prompt_dict):
    if do_train:
        curr_dict = {}
        for name in prompt_names_per_task[task]:
            out = prompt_dict[task+":"+name].apply(example)
            if out==[""]:
                continue
            input_, output_ = out
            curr_dict["inst:"+task+":"+name] = {"task": "inst:"+task+":"+name, "input": input_, "output": output_, "options": []}
        return curr_dict
    input_, output_ = prompt_dict[task].apply(example)
    options = prompt_dict[task].get_answer_choices_list(example)

    # these are for special cases where prompt does not handle answer options properly
    if task=="commonsense_qa":
        assert options is None
        options = example["choices"]["text"]
        assert output_ in options
    elif task=="codah":
        assert options is None
        output_ = output_.strip()
        options = [o.strip() for o in example["candidate_answers"]]
        assert output_ in options, (output_, options)
    elif task=="yelp_polarity":
        assert options == ["no", "yes"] and output_ in ["yes.", "no."], (output_, options)
        output_ = output_[:-1]
        assert output_ in options, (output_, options)
    elif task=="sick":
        assert options is None
        options = ["entailment", "neutral", "contradiction"]
        assert output_ in options, (output_, options)

    if options is None or len(options)==0:
        assert do_train
        options = []
    else:
        assert output_ in options, (task, output_, options)
    return json.dumps({"task": "inst:"+task, "input": input_, "output": output_, "options": options})

def map_hf_dataset_to_list(task, dataset, split, do_train):
    if do_train and "train" not in split:
        return []

    if task=="circa":
        lines = []
        for example in dataset[split]:
            # to be consist with original crossfit script
            example["goldstandard1"] = example["goldstandard2"]
            if example["goldstandard2"] == -1:
                continue
            lines.append(example)
        return lines

    return [e for e in dataset[split]]

def preprocess(dataset, line, config):
    input_, output_ = line
    input_ = input_.strip().replace("\\n", " ")
    output_ = str(output_).split("\t")[0].strip()

    if dataset=="superglue-multirc" and output_=="NO ANSWER!":
        return None

    do_handle_sep = dataset.startswith("race-") or \
            dataset in ["sciq", "social_i_qa", "wiqa", "quail",
                        "superglue-multirc"]

    if do_handle_sep:
        assert input_.count("[SEP]")==1
        input_, context = input_.split("[SEP]")

    alphabet_options = list(string.ascii_uppercase)
    if dataset in ["quail", "quarel"]:
        alphabet_options = ["(" + option + ")" for option in alphabet_options]
    else:
        alphabet_options = [" (" + option + ") " for option in alphabet_options]

    def get_sentences(options):
        sentences = []
        text = input_
        for option in options:
            if option not in text:
                break
            text1, text = text.split(option, 1)
            sentences.append(text1)
        sentences.append(text)
        return sentences

    options = []
    if config["task_type"]=="multi-choice":
        sentences = get_sentences(alphabet_options)

        if len(sentences)>1:
            sentences = [s.strip() for s in sentences]
            input_ = sentences[0]
            options = sentences[1:]

            if dataset=="quarel":
                for i, o in enumerate(options):
                    if o.endswith(" or"):
                        options[i] = o[:-3]
                    if o.endswith("."):
                        options[i] = o[:-1]
                    if output_.endswith("."):
                        output_ = output_[:-1]

                if output_ not in options and output_ + "or" in options:
                    output_ = output_ + "or"

                if output_ not in options and output_.endswith(" or") and output_[:-3] in options:
                    output_ = output_[:-3]

                if output_ not in options and output_=="ext to construction site":
                    output_ = "n" + output_

                if output_ not in options and output_=="oilet paper":
                    output_ = "t" + output_

                if output_ not in options and output_=="aster":
                    output_ = "f" + output_

            if dataset=="superglue-multirc" and len(options)==1:
                return None

            assert len(options)>=2, (dataset, line)
            assert not any(["[SEP]" in option for option in options]), (dataset, line)
            assert output_ in options, (dataset, options, line)

        if len(options)==0 and dataset=="ai2_arc":
            sentences = get_sentences([" (" + str(i) + ") " for i in range(1, 100)])

            if len(sentences)>1:
                sentences = [s.strip() for s in sentences]
                input_ = sentences[0]
                options = sentences[1:]
                assert len(options)>=2, (dataset, line)
                assert not any(["SEP" in option for option in options]), (dataset, line)
                assert output_ in options, (dataset, line)

    elif config["task_type"]=="classification":
        assert len(config["options"])>=2
        options = config["options"]

    if do_handle_sep:
        input_ = context + input_

    return json.dumps({"task": dataset, "input": input_, "output": output_, "options": options})

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def download_from_google_drive(_id, dest):

    if os.path.exists(dest):
        print ("[Already exists] Skipping", dest)
        return

    if "/" in dest:
        dest_dir = "/".join(dest.split("/")[:-1])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    else:
        dest_dir = "."

    command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt""" % (_id, _id, dest)

    ret_code = subprocess.run([command], shell=True)
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(dest))
    else:
        print("Download {} ... [Success]".format(dest))

    if dest.endswith(".zip"):
        command = """unzip %s -d %s && rm %s""" % (dest, dest_dir, dest)

        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Unzip {} ... [Failed]".format(dest))
        else:
            print("Unzip {} ... [Success]".format(dest))



