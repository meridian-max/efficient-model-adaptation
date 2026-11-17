from __future__ import annotations

import copy
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from .lora import LoRALinear, parameter_counts


def digest(x, y):
    return hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()


def make_data(seed, features=32, classes=6, shift_rank=2, train_size=1024, validation_size=256, test_size=512):
    if min(features, classes, shift_rank, train_size, validation_size, test_size) < 1:
        raise ValueError("All dimensions and split sizes must be positive")
    generator = torch.Generator().manual_seed(seed)
    source_weight = torch.randn(classes, features, generator=generator) / features ** .5
    u = torch.randn(classes, shift_rank, generator=generator)
    v = torch.randn(shift_rank, features, generator=generator)
    target_weight = source_weight + .35 * (u @ v) / shift_rank ** .5
    splits = {}
    for offset, (name, size, teacher) in enumerate([
        ("pretrain", train_size * 2, source_weight),
        ("train", train_size, target_weight),
        ("validation", validation_size, target_weight),
        ("test", test_size, target_weight),
    ], 1):
        split_gen = torch.Generator().manual_seed(seed * 1009 + offset)
        x = torch.randn(size, features, generator=split_gen)
        y = (x @ teacher.T).argmax(-1)
        splits[name] = (x, y)
    row_sets = [set(hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in data[0]) for data in splits.values()]
    for i, left in enumerate(row_sets):
        for right in row_sets[i + 1:]:
            if left.intersection(right):
                raise ValueError("Duplicate examples cross dataset splits")
    manifest = {name: {"rows": len(x), "sha256": digest(x, y)} for name, (x, y) in splits.items()}
    return splits, manifest


def fit(model, train, validation, *, steps=160, learning_rate=.03):
    if steps < 1 or not 0 < learning_rate <= 1:
        raise ValueError("Positive steps and learning_rate in (0, 1] required")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0)
    best_loss, best_state, best_step = float("inf"), None, 0
    start = time.perf_counter()
    history = []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(train[0]), train[1])
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                validation_loss = nn.functional.cross_entropy(model(validation[0]), validation[1]).item()
            history.append({"step": step, "train_loss": loss.item(), "validation_loss": validation_loss})
            if validation_loss < best_loss:
                best_loss, best_step = validation_loss, step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"seconds": time.perf_counter() - start, "selected_step": best_step,
            "validation_loss": best_loss, "history": history}


def metrics(model, dataset, *, seed, bootstrap_samples=200):
    model.eval()
    with torch.no_grad():
        logits = model(dataset[0])
        probabilities = logits.softmax(-1)
        correct = (logits.argmax(-1) == dataset[1]).float()
        one_hot = nn.functional.one_hot(dataset[1], logits.shape[-1]).float()
        brier = ((probabilities - one_hot) ** 2).sum(-1).mean().item()
        loss = nn.functional.cross_entropy(logits, dataset[1]).item()
        g = torch.Generator().manual_seed(seed)
        sample_ids = torch.randint(len(correct), (bootstrap_samples, len(correct)), generator=g)
        bootstrap = correct[sample_ids].mean(-1)
        interval = torch.quantile(bootstrap, torch.tensor([.025, .975])).tolist()
    return {"accuracy": correct.mean().item(), "accuracy_bootstrap_95_percent": interval,
            "cross_entropy": loss, "multiclass_brier": brier, "test_examples": len(correct)}


def run_seed(seed, *, rank=2, steps=160, features=32, classes=6):
    torch.manual_seed(seed)
    data, manifest = make_data(seed, features=features, classes=classes, shift_rank=rank)
    base = nn.Linear(features, classes)
    # Base-model selection uses a separate portion of source-domain data, not target test labels.
    x, y = data["pretrain"]
    fit(base, (x[:1536], y[:1536]), (x[1536:], y[1536:]), steps=steps)
    frozen = copy.deepcopy(base).requires_grad_(False)
    full = copy.deepcopy(base)
    adapter = LoRALinear(base, rank=rank, alpha=float(rank))
    results = []
    for name, model in [("frozen", frozen), ("full_finetune", full), ("lora", adapter)]:
        training = fit(model, data["train"], data["validation"], steps=steps) if name != "frozen" else None
        results.append({"method": name, **parameter_counts(model),
                        "metrics": metrics(model, data["test"], seed=seed + 999), "training": training})
    with torch.no_grad():
        merge_error = (adapter(data["test"][0]) - adapter.merged()(data["test"][0])).abs().max().item()
    return {"seed": seed, "dataset_manifest": manifest, "methods": results,
            "adapter_merge_max_absolute_error": merge_error}


def run_experiment(output, *, seeds=(7, 19, 31), rank=2, steps=160):
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Use at least one unique seed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    runs = [run_seed(seed, rank=rank, steps=steps) for seed in seeds]
    summary = []
    for name in ["frozen", "full_finetune", "lora"]:
        methods = [next(m for m in run["methods"] if m["method"] == name) for run in runs]
        values = [m["metrics"]["accuracy"] for m in methods]
        summary.append({"method": name, "mean_accuracy": statistics.mean(values),
                        "sample_std_accuracy": statistics.stdev(values) if len(values) > 1 else None,
                        "trainable_parameters": methods[0]["trainable"], "total_parameters": methods[0]["total"]})
    result = {"schema_version": 1, "task": "synthetic domain-shift multiclass classification",
              "limitations": "A small CPU linear-model experiment, not an LLM or production workload. Test intervals describe fixed-model resampling, not training uncertainty.",
              "config": {"seeds": list(seeds), "rank": rank, "steps": steps, "features": 32, "classes": 6},
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": 1},
              "summary": summary, "runs": runs}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
