#!/usr/bin/env bash
module load cuda-10.0

python sbatch.py launch --cmd="sh test_exp4.sh" --name="adroitv1_train_exp4"
