## filename: trainer.py

import os
import sys
from tqdm import tqdm


def train(model, dataloader):
    model.train()
    if hasattr(model, 'global_epoch'):
        model.global_epoch += 1

    metrics = {}
    total_size = len(dataloader.dataset)

    with tqdm(dataloader, desc="Train", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            results = model.train_step(batch)

            for name, value in results.items():
                if name != "batch_size":
                    metrics.setdefault(name, 0.0)
                    metrics[name] += float(value) * results["batch_size"]

            info = {name: f"{value / total_size:.3f}" for name, value in metrics.items()}
            progress_bar.set_postfix(info)

    return {name: value / total_size for name, value in metrics.items()}


def evaluate(model, dataloader):
    model.eval()
    metrics = {}
    total_size = len(dataloader.dataset)

    with tqdm(dataloader, desc="Evaluate", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            results = model.eval_step(batch)

            for name, value in results.items():
                if name != "batch_size":
                    metrics.setdefault(name, 0.0)
                    metrics[name] += float(value) * results["batch_size"]

            info = {name: f"{value / total_size:.3f}" for name, value in metrics.items()}
            progress_bar.set_postfix(info)

    return {name: value / total_size for name, value in metrics.items()}


def fit(model, train_loader, num_epochs, total_epochs=None, valid_loader=None):
    history = {"train": {}, "valid": {}}
    for epoch in range(1, num_epochs + 1):
        train_results = train(model, train_loader)
        train_info = ", ".join([f"{k}:{v:.3f}" for k, v in train_results.items()])

        if hasattr(model, 'global_epoch') and total_epochs is not None:
            epoch_info = f"[{model.global_epoch:3d}/{total_epochs}]"
        else:
            epoch_info = f"[{epoch:3d}/{num_epochs}]"

        for name, value in train_results.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        if valid_loader is not None:
            valid_results = evaluate(model, valid_loader)
            valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in valid_results.items()])

            for name, value in valid_results.items():
                history["valid"].setdefault(name, [])
                history["valid"][name].append(value)
            print(f"{epoch_info} {train_info} | (val) {valid_info}")
        else:
            print(f"{epoch_info} {train_info}")

    return history
