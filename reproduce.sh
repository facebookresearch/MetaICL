#!/bin/sh
task=$1
method=$2
seed=$3
bs=$4
if [[ $method == *"metaicl" || $method == *"multitask-zero" ]] ; then
    checkpoint="checkpoints/${method}/${task}/model.pt"
    out_dir="checkpoints/${method}/${task}"
    if [[ ! -f $checkpoint ]] ; then
        python -m utils.download --checkpoints --setting $task --method $method
    fi
else
    out_dir="checkpoints/lm"
fi
if [[ $method == "metaicl" ]] ; then
    python test.py --task $task --k 16 --split test --seed $seed --use_demonstrations \
    --test_batch_size $bs --method direct --checkpoint $checkpoint --out_dir $out_dir
fi
if [[ $method == "channel-metaicl" ]] ; then
    python test.py --task $task --k 16 --split test --seed $seed --use_demonstrations \
    --test_batch_size $bs --method channel --checkpoint $checkpoint --out_dir $out_dir
fi
if [[ $method == "multitask-zero" ]] ; then
    python test.py --task $task --split test \
    --test_batch_size $bs --method direct --checkpoint $checkpoint --out_dir $out_dir
fi
if [[ $method == "channel-multitask-zero" ]] ; then
    python test.py --task $task --split test \
    --test_batch_size $bs --method channel --checkpoint $checkpoint --out_dir $out_dir
fi
if [[ $method == "zero" ]] ; then
    python test.py --task $task --k 16 --split test \
    --test_batch_size $bs --method direct --out_dir $out_dir --do_zeroshot
fi
if [[ $method == "pmi-zero" ]] ; then
    python test.py --task $task --k 16 --split test --is_null \
    --test_batch_size $bs --method direct --out_dir $out_dir --do_zeroshot
    python test.py --task $task --k 16 --split test --use_calibration \
    --test_batch_size $bs --method direct --out_dir $out_dir --do_zeroshot
fi
if [[ $method == "channel-zero" ]] ; then
    python test.py --task $task --k 16 --split test \
    --test_batch_size $bs --method channel --out_dir $out_dir --do_zeroshot
fi
if [[ $method == "ic" ]] ; then
    python test.py --task $task --k 16 --seed $seed --split test \
    --test_batch_size $bs --method direct --use_demonstrations --out_dir $out_dir --do_zeroshot
fi
if [[ $method == "pmi-ic" ]] ; then
    python test.py --task $task --k 16 --seed $seed --split test --is_null \
    --test_batch_size $bs --method direct --use_demonstrations --out_dir $out_dir --do_zeroshot
    python test.py --task $task --k 16 --seed $seed --split test --use_calibration \
    --test_batch_size $bs --method direct --use_demonstrations --out_dir $out_dir --do_zeroshot
fi
if [[ $method == "channel-ic" ]] ; then
    python test.py --task $task --k 16 --seed $seed --split test \
    --test_batch_size $bs --method channel --use_demonstrations --out_dir $out_dir --do_zeroshot
fi
