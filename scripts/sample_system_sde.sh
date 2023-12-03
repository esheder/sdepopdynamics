#!/bin/bash

shift=$1
args=("$@")
times="${args[*]:4}"

popfeedback -s $((SLURM_ARRAY_TASK_ID + shift)) -n "$2" sde "$4" --path "$3" --times "$times"