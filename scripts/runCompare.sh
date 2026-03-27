#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: ./scripts/run_compare.sh <n> <k> <T>"
    exit 1
fi

N=$1
K=$2
T=$3

echo "Generating input files for n=$N, k=$K, T=$T..."
python3 scripts/generateInput.py --n $N --k $K --T $T

INT_FILE="input/int/dataset_n_${N}_k_${K}_T_${T}.txt"
FLOAT_FILE="input/float/dataset_n_${N}_k_${K}_T_${T}.txt"

if [ -f "bin/compare" ]; then
    echo "Running comparison..."
    ./bin/compare "$INT_FILE" "$FLOAT_FILE"
else
    echo "Error: bin/compare not found."
    exit 1
fi
