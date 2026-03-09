# Code Adapted from HuggingFace: https://huggingface.co/datasets/akhauriyash/Code-Regression

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset # HuggingFace datasets library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scipy.stats import spearmanr
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--repo', type=str, default="akhauriyash/RLM-GemmaS-Code-v0", help="HuggingFace repo id from where we are pulling the model from")
parser.add_argument('--dataset', type=str, default="akhauriyash/Code-Regression", help="HuggingFace dataset path")
parser.add_argument('--split', type=str, default="train", help="Split of the dataset to run model on")
parser.add_argument('--max_items', type=int, default=10000, help="Max # of datapoints to process")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size (how many datapoints to process at once)")
parser.add_argument('--spaces', nargs='+', default=["KBSS", "CDSS", "APPS"], help="Spaces (dataset splits for the Code Metrics dataset to run model on")
parser.add_argument('--top_filter', type=int, default=1, help="Percent by which to filter top % of target values")
parser.add_argument('--bottom-filter', type=int, default=0, help="Percent by which to filter bottom % of target values")
parser.add_argument('--language', type=str, default=None, help="Specify a language to filter the CDSS space on. Model will only run on datapoints with that language")
parser.add_argument('--verbose', type=bool, default=True, help="Output extra logging/debugging statements or not")
parser.add_argument('--save', type=bool, default=True, help="Save results to a CSV file or not")

# Init args
parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

def load_dataset():
    # Load dataset
    if args.dataset.endswith('.csv'):
        # Load CSV dataset
        dataset = load_dataset('csv', data_files=args.dataset, split=args.split)
    else:
        # Load directly from HuggingFace
        dataset = load_dataset(args.dataset, split=args.split)
    return dataset

def process_dataset(dataset, SPACE, max_items, language):
    inputs, targets = [], []
    for row in tqdm(dataset, desc=f"Processing {SPACE} till {max_items} items"):
        if row.get("space") == SPACE and "input" in row and "target" in row:
            try:
                lang = eval(row['metadata'])['language'] if SPACE == "CDSS" else None
                if SPACE != "CDSS" or language is None or lang == language:
                    targets.append(float(row["target"])) # Convert target metric to float
                    if SPACE == "CDSS":
                        inputs.append(f"# {SPACE}\n# Language: {lang}\n{row['input']}")
                    else:
                        inputs.append(f"{SPACE}\n{row['input']}")
            except: continue
            if len(inputs) >= args.max_items: break # Stop processing dataset once we hit max_items
    return inputs, targets

def run_model(model, tok, inputs, batch_size, device, n_out_tokens):
    preds = []
    # Iterate through each batch of data
    for i in tqdm(range(0, len(inputs), batch_size)):
        # Tokenize inputs
        enc = tok(inputs[i:i+batch_size], return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
        batch_preds = []
        for _ in range(8): # Ask the model to do the same question 8 times
            # Generate new tokens
            out = model.generate(**enc, max_new_tokens=n_out_tokens, min_new_tokens=n_out_tokens, do_sample=True, top_p=0.95, temperature=1.0)
            # Convert tokens back into numbers (floats)
            decoded = [tok.token_ids_to_floats(seq.tolist()) for seq in out]
            decoded = [d[0] if isinstance(d, list) and d else float("nan") for d in decoded] # Get first # from output, nan is no number
            batch_preds.append(decoded) # Add to current batch preds array
        preds.extend(torch.tensor(batch_preds).median(dim=0).values.tolist()) # Find median of the 8 guesses and add to final predictions array

    return preds

def filter_dataset(targets_arr, preds_arr, top_filter, bottom_filter):
    mask = np.ones(len(targets_arr), dtype=bool) # Init mask
    if top_filter > 0:
        upper_limit = np.percentile(targets_arr, 100 - top_filter)
        mask &= (targets_arr <= upper_limit)
        print(f"Filtering out the top {top_filter}% of data (Limit: {upper_limit})", flush=True)
    if bottom_filter > 0: 
        lower_limit = np.percentile(targets_arr, bottom_filter)
        mask &= (targets_arr >= lower_limit)
        print(f"Filtering out the bottom {bottom_filter}% of data (Limit: {lower_limit})", flush=True)
    # Apply mask
    targets_filtered = targets_arr[mask]
    preds_filtered = preds_arr[mask]

    return targets_filtered, preds_filtered


def main():
    dataset = load_dataset() # Load dataset

    ####################################
    #### DEFINE LOSS WEIGHTING HERE ####
    ####################################

    # Init model
    tok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True) # Break up strings into tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.repo, trust_remote_code=True).to(device).eval() # Define model, in eval mode
    results = {} # Empty dict to store results in for later
    # Calculate # of tokens model is expected to output
    n_out_tokens = getattr(model.config, "num_tokens_per_obj", 8) * getattr(model.config, "max_num_objs", 1)
    n_out_tokens = model.config.num_tokens_per_obj * model.config.max_num_objs

    print(f"Processing {args.max_items} rows of data per space")

    for SPACE in args.spaces:
        # Process dataset
        inputs, targets = process_dataset(dataset, SPACE, args.max_items, args.language)
        # Run model
        preds = run_model(model, tok, inputs, args.batch_size, device, n_out_tokens)
        
        targets_arr = np.array(targets) # Convert targets list to Numpy array
        preds_arr = np.array(preds) # Convert preds list to Numpy array

        # Filter out dataset if defined in args
        if args.top_filter > 0 or args.bottom_filter > 0 :
            targets_arr, preds_arr = filter_dataset(targets_arr, preds_arr, args.top_filter, args.bottom_filter)

        # Show sample predictions for logging/debugging
        if args.verbose:
            print(f"\n  Sample predictions vs targets:", flush=True)
            for i in range(min(5, len(preds_arr))):
                diff = preds_arr[i] - targets_arr[i]
                print(f"    [{i}] Pred: {preds_arr[i]:.4f}, Target: {targets_arr[i]}, Diff: {diff}", flush=True)

        # Calculate differences
        differences = preds_arr - targets_arr
        abs_differences = np.abs(differences)

        if args.verbose:
            print(f"\n  Prediction statistics:", flush=True)
            print(f"    Predictions - Min: {np.min(preds_arr):.4f}, Max: {np.max(preds_arr):.4f}, Mean: {np.mean(preds_arr):.4f}", flush=True)
            print(f"    Targets     - Min: {np.min(targets_arr):.4f}, Max: {np.max(targets_arr):.4f}, Mean: {np.mean(targets_arr):.4f}", flush=True)
            print(f"    Differences - Min: {np.min(differences):.4f}, Max: {np.max(differences):.4f}, Mean: {np.mean(differences):.4f}", flush=True)
            print(f"    Abs Diffs   - Min: {np.min(abs_differences):.4f}, Max: {np.max(abs_differences):.4f}", flush=True)

        # Calculate Spearman correlation
        spear, _ = spearmanr(targets_arr, preds_arr)
        # Calculate MAE
        mae = np.mean(abs_differences)

        ########################################
        ### CALCULATE/ADD WEIGHTED LOSS HERE ###
        ########################################

        # Save to results
        results[SPACE] = {'spearman': spear, 'mae': mae}

        if args.verbose:
            print(f"\n{SPACE} - Spearman ρ: {spear:.3f}, MAE: {mae:.4f}", flush=True)

        # Save results to a file so we can plot in the future
        if args.save: 
            preds_data = []
            for t, p, diff, abs_diff in zip(targets_arr, preds_arr, differences, abs_differences):
                preds_data.append({
                    "Space": SPACE,
                    "Target": t,
                    "Prediction": p,
                    "Error": diff,
                    "Absolute_Error": abs_diff
                })
            preds_df = pd.DataFrame(preds_data)
            preds_df.to_csv("prediction_data.csv", index=False)


#### PRINT OUT RESULTS ####
    print("\nSummary of Results:")
    print("="*60)
    print(f"{'Space':<15} {'Spearman ρ':<15} {'MAE':<10}")
    print("-"*60)
    for space in args.spaces:
        print(f"{space:<15} {results[space]['spearman']:<15.3f} {results[space]['mae']:<10.4f}")
    print("="*60)


if __name__ == '__main__':
    main()