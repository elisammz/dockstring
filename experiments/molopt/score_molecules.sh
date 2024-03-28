#!/usr/bin/env bash

# Activate your Python environment if needed
# source /path/to/your/env/bin/activate

# Install Python dependencies if needed
# pip install pandas dockstring

# Run the Python script
python - <<END
import pandas as pd
from dockstring import load_dataset, dock_molecules

# Load the generated molecules
generated_mols = []
for trial in range(3):
    trial_path = f"./results/molopt/bo_gp_exact/GFR/trial-{trial}.json"
    trial_data = pd.read_json(trial_path)
    generated_mols.extend(trial_data['smiles'].tolist())

# Load the dataset with other objectives
dataset = load_dataset("dockstring-dataset.tsv")

# Compute scores for the generated molecules against other objectives
other_objectives = ["EGFR", "IGF1R"]
scores = dock_molecules(generated_mols, dataset, other_objectives)

# Analyze and compare the scores
for obj in other_objectives:
    print(f"Scores for {obj}:")
    print(scores[obj])
END