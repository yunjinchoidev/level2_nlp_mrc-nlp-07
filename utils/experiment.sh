#!/bin/bash
# Define arrays of b and k1 values to experiment with

b_values=(0.6 0.7 0.8 0.9 1.0)
k1_values=(1.1 1.2 1.3 1.4 1.5)

# Iterate over all combinations of b and k1
for b in "${b_values[@]}"; do
    for k1 in "${k1_values[@]}"; do
        python bm25_experiment.py $b $k1
    done
done