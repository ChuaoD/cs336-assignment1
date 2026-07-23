from __future__ import annotations

from cs336_basics.model import TransformerLM
from cs336_basics.utils import (
    cross_entropy_loss,
    gradient_clipping,
)
from cs336_basics.optimizer import (
    AdamW,
    lr_cosine_schedule
)

from cs336_basics.data_loader import load_data
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

import torch
import numpy as np
import wandb
import argparse

import json
from pathlib import Path

def train(config_path: Path):

    # Read configuration
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    
    # Load train and valididation dataset
    data_config = config["data"]
    train_path = data_config["train_path"]
    validation_path = data_config["validation_path"]
    train_data = np.load(train_path, mmap_mode="r")
    validation_data = np.load(validation_path, mmap_mode="r")

    # Construct model and optimizer
    context_length = config["model"]["context_length"]
    model = TransformerLM(**config["model"])
    optimizer_config = config["optimizer"].copy()
    optimizer_config["betas"] = tuple(optimizer_config["betas"])
    optimizer = AdamW(model.parameters(), **optimizer_config)
    lr = optimizer_config["lr"]

    # Read learning rate scheduler configuration
    scheduler_config = config["scheduler"]

    # Read training configuration
    training_config = config["training"]
    batch_size = training_config["batch_size"]
    max_iters = training_config["max_iters"]
    gradient_clip_max_norm = training_config["gradient_clip_max_norm"]
    device = training_config["device"]

    start_iter = 0
    best_validation_loss = float("inf")

    # Restart from the checkpoint if any
    checkpoint = config["checkpoint"]["resume_from"]
    checkpoint_interval = config["checkpoint"]["interval"]
    checkpoint_out_dir = config["checkpoint"]["output_dir"]
    if checkpoint:
        start_iter = load_checkpoint(checkpoint, model, optimizer) + 1

    # Read evaluation configuration
    eval_interval = config["evaluation"]["interval"]
    num_eval_batches = config["evaluation"]["num_eval_batches"]

    # Read logging configuration
    logging_config = config["logging"]
    logging_interval = logging_config["interval"]
    wandb_mode = (
    logging_config["wandb_mode"]
    if logging_config["use_wandb"]
    else "disabled"
    )
    wandb.init(
        project=logging_config["project"],
        name=logging_config["run_name"],
        config={
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_iters": max_iters  
        }, 
        mode=wandb_mode,
    )

    train_loss = 0
    num_log_batches = 0

    for iter in range(start_iter, max_iters):
        model.train()
        # Determine learning rate
        current_lr = lr_cosine_schedule(iter + 1, **config["scheduler"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        # Load data batch
        inputs, targets = load_data(train_data, batch_size, context_length, device)
        # Clear gradients from the previous step
        optimizer.zero_grad()
        # Forward pass
        logits = model(inputs)
        # Compute loss
        loss = cross_entropy_loss(logits, targets)
        # Compute gradients
        loss.backward()
        # Clip gradient
        gradient_clipping(
            model.parameters(), 
            max_l2_norm=gradient_clip_max_norm
        )
        # Update parameters
        optimizer.step()

        train_loss += loss.item()
        num_log_batches += 1
        completed_iters = iter + 1

        # Periodic logging
        if completed_iters % logging_interval == 0:
            avg_train_loss = train_loss / num_log_batches
            print(f"\nCompleted training {completed_iters} iterations.")
            print(f"\nAverage training loss: {avg_train_loss:.4f}.")
            print(f"\nNumber of tokens processed: {batch_size * context_length}.\n")
            wandb.log({
                "iter": completed_iters,
                "train/avg_loss": avg_train_loss,
                "train/learning_rate": current_lr,
            })
            train_loss = 0
            num_log_batches = 0

        # Periodic evaluation
        if completed_iters % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                for _ in range(num_eval_batches):
                    valid_inputs, valid_targets = load_data(validation_data, batch_size, context_length, device)

                    valid_logits = model(valid_inputs)
                    valid_loss = cross_entropy_loss(valid_logits, valid_targets)
                    total_valid_loss += valid_loss.item()
                avg_valid_loss = valid_loss / num_eval_batches
                print(f"\nEvaluation after {iter + 1} iterations.")
                print(f"\nAverage evaluation loss on {num_eval_batches} batches: {avg_valid_loss:.4f}.\n")
                wandb.log({
                    "iter": completed_iters,
                    "valid/avg_loss": avg_valid_loss
                })
                if avg_valid_loss < best_validation_loss:
                    best_validation_loss = avg_valid_loss
                    save_checkpoint(model, optimizer, iter, f"{checkpoint_out_dir}/best.pt")

        # Periodic checkpoints
        if (iter + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iter, f"{checkpoint_out_dir}/{iter}.pt")

    save_checkpoint(model, optimizer, iter, f"{checkpoint_out_dir}/step_{iter}.pt")

    

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train a language model."
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration file"
    )

    args = parser.parse_args()

    train(args.config_path)

if __name__ == "__main__":
    main()
