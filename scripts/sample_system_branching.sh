#!/bin/bash

shift=$1
args=("$@")
times="${args[*]:3}"

popfeedback -s $((SLURM_ARRAY_TASK_ID + shift)) -n "$2" branching --path "$3" --times "$times"