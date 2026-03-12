# Code Adapted from HuggingFace: https://huggingface.co/datasets/akhauriyash/Code-Regression

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from datasets import load_dataset # HuggingFace datasets library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scipy.stats import spearmanr
from tqdm import tqdm
import argparse
from dataloader import CDSSDataset, CDSSLDS, CDSSMDS
import os

'''
Helper Functions
'''
def get_dataset(dataset, split):
    # Load dataset
    if dataset.endswith('.csv'):
        # Load CSV dataset
        dataset = load_dataset('csv', data_files=dataset, split=split)
    else:
        # Load directly from HuggingFace
        dataset = load_dataset(dataset, split=split)
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
            if len(inputs) >= max_items: break # Stop processing dataset once we hit max_items
    return inputs, targets

def filter_top_percentile(inputs, targets, top_percent=1.0):
    targets_arr = np.array(targets)
    q_threshold = np.quantile(targets_arr, (100 - top_percent) / 100)
    mask = targets_arr <= q_threshold
    
    filtered_inputs = [input for input, keep in zip(inputs, mask) if keep]
    filtered_targets = [target for target, keep in zip(targets, mask) if keep]
    
    print(f"Filtered top {top_percent}%: kept {len(filtered_inputs)}/{len(inputs)}")
    return filtered_inputs, filtered_targets

def get_selected_indices(dataset, SPACE, max_items, language):
    '''For CDSSLDS/CDSSMDS - returns indices instead of inputs/targets'''
    selected_indices = []
    for idx, row in enumerate(dataset):
        if len(selected_indices) >= max_items:
            break
        if row.get("space") == SPACE and "input" in row and "target" in row:
            try:
                lang = eval(row['metadata'])['language'] if SPACE == "CDSS" else None
                if SPACE != "CDSS" or language is None or lang == language:
                    selected_indices.append(idx)
            except:
                continue
    return selected_indices

# Train model
def train_model(model, train_loader, optimizer, scheduler, device, num_epochs, save_path="./checkpoints"):
    '''
    Train the model with option to weight the losses
    '''
    # Use model's training mode
    model.train()
    # Create save path
    os.makedirs(save_path, exist_ok=True)
    # Iterate model through epochs
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        # Iterate through batches
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            weights = batch['weight'].to(device)
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            # Get loss and apply sample weights
            loss = outputs.loss

            weighted_loss = (loss * weights).mean()
            
            # Backward pass
            optimizer.zero_grad()
            weighted_loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()
            scheduler.step()

            # Track loss
            total_loss += weighted_loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{weighted_loss.item()}',
                'avg_loss': f'{total_loss/num_batches}'
            })
        
        # Summarize epoch
        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"{save_path}/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    print("\nTraining complete!")
    return model

def main():
    # Add training arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=500, help="Warmup steps for scheduler")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument('--repo', type=str, default="akhauriyash/RLM-GemmaS-Code-v0", help="HuggingFace repo id from where we are pulling the model from")
    # Data-related arguments
    parser.add_argument('--dataset', type=str, default="akhauriyash/Code-Regression")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--max_items', type=int, default=10000, help="Max # of items to process for training dataset")
    parser.add_argument('--max_eval_items', type=int, default=1024, help="Max # of items to evaluate")
    parser.add_argument('--space', type=str, default="CDSS", help="Type of space to use from dataset (i.e. CDSS, APPS, KBSS in the Code Metrics dataset)")
    parser.add_argument('--language', type=str, default=None, help="Specifically for CDSS, the language of the input code. Also the attribute a to look at.")
    parser.add_argument('--batch_size', type=int, default=16)
    # Dataset type (based on weighting)
    parser.add_argument('--dataset_type', type=str, default='raw',
                   choices=['raw', 'lds', 'mds'],
                   help="Choose: raw, lds, or mds")
    # Filtering
    parser.add_argument('--filter_top_percent', type=float, default=1.0, help="How much % to filter original dataset's target metric values on")

    # Parse args
    parser.set_defaults(augment=True)
    args, unknown = parser.parse_known_args()

    # Load dataset
    dataset = get_dataset(args.dataset, args.split)

    # Init model
    tok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True) # Break up strings into tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.repo, trust_remote_code=True).to(device)

    # Create dataset based on type input
    if args.dataset_type == 'raw':
        inputs, targets = process_dataset(dataset, args.space, args.max_items, args.language)
        inputs, targets = filter_top_percentile(inputs, targets, args.filter_top_percent)
        # Use raw dataset class
        train_dataset = CDSSDataset(inputs, targets, tok)
        
    elif args.dataset_type == 'lds':
        indices = get_selected_indices(dataset, args.space, args.max_items, args.language)
        # Use CDSSLDS class
        train_dataset = CDSSLDS(dataset, indices)
        
    elif args.dataset_type == 'mds':
        indices = get_selected_indices(dataset, args.space, args.max_items, args.language)
        # Use CDSSMDS class
        train_dataset = CDSSMDS(dataset, indices, tok)

    print(f"Total training samples: {len(train_dataset)}")

    # Created dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Set up optimizer and scheduler
    # Use AdamW optimizer for simplicity
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # Learning rate scheduler (linear warmup then decay)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Train model
    print("\n" + "#"*50)
    print("Starting training!")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        save_path=args.checkpoint_dir
    )

    # Evaluate model
    print("\n" + "#"*50)
    print("Starting evaluation!")
    trained_model.eval()

    eval_inputs, eval_targets = process_dataset(dataset, args.space, args.max_eval_items, args.language)
    if args.filter_top_percent > 0:
        eval_inputs, eval_targets = filter_top_percentile(eval_inputs, eval_targets, args.filter_top_percent)

    preds = []
    print(f"Generating predictions for {len(eval_inputs)} samples")

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_inputs), args.batch_size)):
            batch_inputs = eval_inputs[i:i+args.batch_size]
            enc = tok(batch_inputs, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
            
            # Generate predictions
            out = model.generate(**enc, max_new_tokens=8, do_sample=False)
            decoded = [tok.token_ids_to_floats(seq.tolist()) for seq in out]
            decoded = [d[0] if isinstance(d, list) and d else float("nan") for d in decoded]
            preds.extend(decoded)
    
        # Calculate metrics
        preds_arr = np.array(preds)
        targets_arr = np.array(eval_targets[:len(preds)])
    
        # Remove NaNs
        valid_mask = ~np.isnan(preds_arr)
        preds_arr = preds_arr[valid_mask]
        targets_arr = targets_arr[valid_mask]
        
        mae = np.mean(np.abs(preds_arr - targets_arr))
        spearman_corr, _ = spearmanr(preds_arr, targets_arr)
    
    print(f"MAE: {mae}")
    print(f"Spearman correlation: {spearman_corr}")


if __name__ == '__main__':
    main()