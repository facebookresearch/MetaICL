import os
import json
import numpy as np
import pickle as pkl

task = "financial_phrasebank"

def compare_task(task):

    def compare_checkpoints(checkpoint1, checkpoint2):

        with open(checkpoint1.format(task), "rb") as f:
            loss1 = pkl.load(f)

        with open(checkpoint2.format(task if task.startswith("inst") else "cf-"+task), "rb") as f:
            loss2 = pkl.load(f)

        loss1 = np.array(loss1)
        loss2 = np.array(loss2)
        #print (loss1.shape, loss2.shape)
        return np.sum(np.abs(loss1-loss2))<1 #e-05

    checkpoint1 = "/checkpoint/sewonmin/GPT2/checkpoints-test-final/lm/{}-test-null-direct.pkl"
    checkpoint2 = "/private/home/sewonmin/GPT2/out/gpt2-large/discriminative-2-as/s_{}/None-t=0.pkl"
    return compare_checkpoints(checkpoint1, checkpoint2)
    b_loss1, b_loss2 = compare_checkpoints(checkpoint1, checkpoint2)

    checkpoint1 = "/checkpoint/sewonmin/GPT2/checkpoints-test-final/lm/{}-test-direct.pkl"
    checkpoint2 = "/private/home/sewonmin/GPT2/out/gpt2-large/discriminative-2-as/s_{}/test-t=0.pkl"
    loss1, loss2 = compare_checkpoints(checkpoint1, checkpoint2)

    return

    with open("data/{}/{}_16_13_test.jsonl".format(task, task), "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    options = data[0]["options"]
    assert np.all([dp["options"]==options for dp in data[1:]])
    gold_labels = [options.index(dp["output"]) for dp in data]

    def get_predictions(loss, b_loss):
        loss -= b_loss
        assert len(data)*3==len(loss)
        predictions = []
        for i, dp in enumerate(data):
            curr_loss = loss[3*i:3*(i+1)]
            pred_idx = np.argmin(curr_loss)
            predictions.append(pred_idx)
        return predictions

    predictions = get_predictions(loss1, b_loss1)
    print (np.mean(np.array(predictions)==np.array(gold_labels)))

    from metaicl.data import MetaICLData
    m_data = MetaICLData(method="direct", use_demonstrations=False)
    m_data.tensorize([], data)
    acc = m_data.evaluate([options[i] for i in predictions], [options[i] for i in gold_labels],
                        is_classification=True)
    print (acc)
    from IPython import embed; embed()


for config_postfix in ["", "_noinst", "_inst"]:
    print ("-"*30, config_postfix, "-"*30)
    with open("config/hr_to_lr%s.json" % config_postfix, "r") as f:
        tasks = json.load(f)["test"]
    for task in tasks:
        c = compare_task(task)
        print (task, c)


