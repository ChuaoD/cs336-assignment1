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
import time

import json
from pathlib import Path

def save_best_loss(best_loss: float, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({"best_validation_loss": best_loss}, f)


def load_best_loss(path: Path) -> float:
    if not path.exists():
        return float("inf")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return float(data["best_validation_loss"])

def train(config_path: Path, learning_rate: float | None = None):

    # Read configuration
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if learning_rate is not None:
        config["optimizer"]["lr"] = learning_rate
        config["scheduler"]["max_learning_rate"] = learning_rate
        config["scheduler"]["min_learning_rate"] = learning_rate * 0.1

    # Load train and valididation dataset
    data_config = config["data"]
    train_path = data_config["train_path"]
    validation_path = data_config["validation_path"]
    dtype = data_config["dtype"]
    train_data = np.memmap(train_path, dtype=dtype, mode="r")
    validation_data = np.memmap(validation_path, dtype=dtype, mode="r")

    # Read training configuration
    training_config = config["training"]
    batch_size = training_config["batch_size"]
    max_iters = training_config["max_iters"]
    gradient_clip_max_norm = training_config["gradient_clip_max_norm"]
    device = training_config["device"]

    # Construct model and optimizer
    context_length = config["model"]["context_length"]
    model = TransformerLM(**config["model"], device=device)
    optimizer_config = config["optimizer"].copy()
    optimizer_config["betas"] = tuple(optimizer_config["betas"])
    optimizer = AdamW(model.parameters(), **optimizer_config)
    lr = optimizer_config["lr"]

    start_iter = 0

    # Restart from the checkpoint if any
    checkpoint = config["checkpoint"]["resume_from"]
    checkpoint_interval = config["checkpoint"]["interval"]
    checkpoint_out_dir = (
        Path(config["checkpoint"]["output_dir"]) / f"lr-{lr:g}"
    )
    checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_loss_path = checkpoint_out_dir / "best_loss.json"
    if checkpoint:
        start_iter = load_checkpoint(checkpoint, model, optimizer) + 1
        best_validation_loss = load_best_loss(best_loss_path)

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
        name=f"{logging_config['run_name']}-lr-{lr:g}",
        config={
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_iters": max_iters  
        }, 
        mode=wandb_mode,
    )

    total_train_loss = 0
    num_log_batches = 0

    start_time = time.perf_counter()

    for iter in range(start_iter, max_iters):
        model.train()
        # Determine learning rate
        current_lr = lr_cosine_schedule(iter, **config["scheduler"])
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

        total_train_loss += loss.item()
        num_log_batches += 1
        completed_iters = iter + 1

        # Periodic logging
        if completed_iters % logging_interval == 0:
            avg_train_loss = total_train_loss / num_log_batches
            elapsed_time = time.perf_counter() - start_time
            print(f"\nCompleted training {completed_iters} iterations.")
            print(f"\nElapsed time: {elapsed_time}.")
            print(f"\nAverage training loss: {avg_train_loss:.4f}.")
            print(f"\nNumber of tokens processed: {num_log_batches * batch_size * context_length}.\n")
            wandb.log({
                "iter": completed_iters,
                "train/avg_loss": avg_train_loss,
                "train/learning_rate": current_lr,
                "train/elapsed_time": elapsed_time
            })
            total_train_loss = 0
            num_log_batches = 0

        # Periodic evaluation
        if completed_iters % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                total_valid_loss = 0
                for _ in range(num_eval_batches):
                    valid_inputs, valid_targets = load_data(validation_data, batch_size, context_length, device)

                    valid_logits = model(valid_inputs)
                    valid_loss = cross_entropy_loss(valid_logits, valid_targets)
                    total_valid_loss += valid_loss.item()
                avg_valid_loss = total_valid_loss / num_eval_batches
                elapsed_time = time.perf_counter() - start_time
                print(f"\nEvaluation after {iter + 1} iterations.")
                print(f"\nElapsed time: {elapsed_time}.")
                print(f"\nAverage evaluation loss on {num_eval_batches} batches: {avg_valid_loss:.4f}.\n")
                wandb.log({
                    "iter": completed_iters,
                    "valid/avg_loss": avg_valid_loss,
                    "valid/elapsed_time": elapsed_time
                })
                if avg_valid_loss < best_validation_loss:
                    best_validation_loss = avg_valid_loss
                    save_checkpoint(model, optimizer, iter, f"{checkpoint_out_dir}/best.pt")
                    save_best_loss(best_validation_loss, best_loss_path)


        # Periodic checkpoints
        if (iter + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iter, f"{checkpoint_out_dir}/step_{completed_iters}.pt")

    

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train a language model."
    )

    parser.add_argument(
        "--config_path",
        type=Path,
        help="Path to the configuration file"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    train(args.config_path, args.learning_rate)

if __name__ == "__main__":
    main()
