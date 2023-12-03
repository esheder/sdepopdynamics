#!/bin/bash

shift=$1
args=("$@")
times="${args[*]:4}"
seed=$((SLURM_ARRAY_TASK_ID + shift))

popfeedback -s "$seed" -n "$2" sde "$4" --path "$3" --times "$times" >result_sde_"$seed".out