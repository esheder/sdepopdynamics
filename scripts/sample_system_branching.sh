#!/bin/bash

shift=$1
args=("$@")
times="${args[*]:3}"
seed=$((SLURM_ARRAY_TASK_ID + shift))

popfeedback -s "$seed" -n "$2" branching --path "$3" --times "$times" >result_branching_"$seed".out