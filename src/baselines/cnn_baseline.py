"""Untuned reference CNN for the learning-curve comparison.

WHY UNTUNED: the comparison is only credible if neither side was tuned to
make the other look good or bad. The architecture and training protocol are
textbook defaults fixed in ``config.CNN_BASELINE`` before any result was seen,
and are applied unchanged at every curve point. No architecture search, no
hyperparameter search, no augmentation (the classical pipeline gets none).

Fairness measures:
  * identical seeded splits and the identical fixed test set as the classical
    models (``data_loader.stratified_split``)
  * identical input: the preprocessed 64x64 binary images
  * early stopping uses a validation slice carved from the TRAINING set only;
    the model is then refit on 100 % of the training set for the chosen
    number of epochs, so it sees exactly the images the classical models see
  * metrics computed by the shared ``learning_curve.score_predictions``

This module is never imported by the pipeline or the app.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import nn

from src import config
from src.utils import get_logger

log = get_logger("cnn_baseline")
C = config.CNN_BASELINE


def seed_everything(seed: int = config.SEED) -> None:
    """Make CPU training bit-for-bit reproducible."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    threads = C.num_threads or os.cpu_count() or 1
    torch.set_num_threads(threads)


class PlainCNN(nn.Module):
    """3 x [conv3x3 -> BN -> ReLU -> maxpool2] -> dropout -> dense -> ReLU -> dropout -> dense."""

    def __init__(self, n_classes: int, image_size: int = config.PREPROCESS.image_size) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for c_out in C.conv_channels:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        side = image_size // (2 ** len(C.conv_channels))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(C.dropout), nn.Linear(c_in * side * side, C.dense_units),
            nn.ReLU(inplace=True), nn.Dropout(C.dropout), nn.Linear(C.dense_units, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.features(x))


def describe_architecture(n_classes: int) -> str:
    """Human-readable architecture string for the report."""
    convs = " -> ".join(f"conv3x3({c})-BN-ReLU-pool" for c in C.conv_channels)
    return f"{convs} -> dropout({C.dropout}) -> dense({C.dense_units})-ReLU -> dropout({C.dropout}) -> dense({n_classes})"


def model_size_bytes(model: nn.Module) -> int:
    """Size of the serialised state_dict."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 (N, S, S) ink=255 -> float (N, 1, S, S) in [0, 1]."""
    return torch.from_numpy(images).unsqueeze(1)


def _batches(n: int, batch_size: int, gen: torch.Generator | None) -> list[torch.Tensor]:
    order = torch.randperm(n, generator=gen) if gen is not None else torch.arange(n)
    return list(order.split(batch_size))


def _run_epoch(model: nn.Module, X: torch.Tensor, y: torch.Tensor, opt: torch.optim.Optimizer,
               gen: torch.Generator) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for idx in _batches(len(X), C.batch_size, gen):
        xb = X[idx].float().div_(255.0)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), y[idx])
        loss.backward()
        opt.step()
        total += float(loss) * len(idx)
    return total / len(X)


@torch.no_grad()
def predict_proba(model: nn.Module, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Softmax probabilities for uint8 images."""
    model.eval()
    X = _to_tensor(images)
    out = [torch.softmax(model(X[i: i + batch_size].float().div_(255.0)), 1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def _val_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total = sum(float(loss_fn(model(X[i: i + 512].float().div_(255.0)), y[i: i + 512]))
                for i in range(0, len(X), 512))
    return total / len(X)


def _stratified_val_mask(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Per-class validation slice (at least 1 image per class with >= 2 images)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = int(round(frac * len(idx)))
        k = min(max(k, 1), len(idx) - 1) if len(idx) >= 2 else 0
        mask[rng.choice(idx, size=k, replace=False)] = True
    return mask


def _fit(n_classes: int, X: torch.Tensor, y: torch.Tensor, epochs: int, seed: int,
         X_val: torch.Tensor | None = None, y_val: torch.Tensor | None = None
         ) -> tuple[nn.Module, list[dict[str, float]], int]:
    """Train for up to ``epochs``; with validation data, early-stop on val loss.

    Returns (model with best weights, history, best_epoch).
    """
    seed_everything(seed)
    model = PlainCNN(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=C.learning_rate)
    gen = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    best, best_epoch, best_state, waited = np.inf, epochs, None, 0
    for ep in range(1, epochs + 1):
        t = time.perf_counter()
        tr_loss = _run_epoch(model, X, y, opt, gen)
        rec = {"epoch": ep, "train_loss": round(tr_loss, 5), "seconds": round(time.perf_counter() - t, 2)}
        if X_val is not None:
            vl = _val_loss(model, X_val, y_val)
            rec["val_loss"] = round(vl, 5)
            if vl < best - 1e-4:
                best, best_epoch, waited = vl, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
        history.append(rec)
        log.info("    epoch %2d  train_loss %.4f%s  (%.1fs)", ep, tr_loss,
                 f"  val_loss {rec['val_loss']:.4f}" if "val_loss" in rec else "", rec["seconds"])
        if X_val is not None and waited >= C.early_stopping_patience:
            log.info("    early stop: no val improvement for %d epochs (best epoch %d)", waited, best_epoch)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def train_and_evaluate(train_imgs: np.ndarray, train_labels: np.ndarray, classes: list[str],
                       seed: int = config.SEED) -> tuple[nn.Module, dict[str, Any]]:
    """Fixed protocol: early stopping on a train-only validation slice, then
    (optionally) refit on the full training set for the selected epoch count.

    Returns:
        (trained model, info dict with timings, epochs, history, size)
    """
    if not C.refit_on_full_train:
        log.warning("refit_on_full_train is OFF: the CNN trains on fewer images than the classical models. "
                    "Results from this run must NOT be reported as the comparison.")
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cls_to_idx[c] for c in train_labels], dtype=np.int64)
    X_all = _to_tensor(train_imgs)
    val = _stratified_val_mask(y_all, C.val_fraction, seed)

    t0 = time.perf_counter()
    log.info("  phase 1: early stopping (train %d / val %d)", int((~val).sum()), int(val.sum()))
    model, hist1, best_epoch = _fit(len(classes), X_all[~val], torch.from_numpy(y_all[~val]), C.max_epochs, seed,
                                    X_all[val], torch.from_numpy(y_all[val]))
    hist2: list[dict[str, float]] = []
    # WHY THIS REFIT EXISTS - do not remove it in a refactor.
    # Phase 1 trains on only (1 - val_fraction) of the training set, because a
    # slice is held out to choose the epoch count. Without this refit, the CNN
    # would learn from ~10 % FEWER images than the classical models at every
    # curve point. That would handicap the CNN and make the comparison invalid
    # IN OUR FAVOUR (classical would look better partly because it saw more
    # data). Phase 1 only picks the epoch count; the reported model is then
    # trained on 100 % of the same training images the classical models use.
    if C.refit_on_full_train:
        log.info("  phase 2: refit on 100%% of training data for %d epochs", best_epoch)
        model, hist2, _ = _fit(len(classes), X_all, torch.from_numpy(y_all), best_epoch, seed)
    train_time = time.perf_counter() - t0
    info = {
        "train_time_s": round(train_time, 2),
        "best_epoch": best_epoch,
        "epochs_phase1": len(hist1),
        "epochs_refit": len(hist2),
        "history_phase1": hist1,
        "history_refit": hist2,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "model_size_bytes": model_size_bytes(model),
        "n_train": int(len(y_all)),
    }
    return model, info


def protocol_description() -> dict[str, Any]:
    """Everything a reviewer needs to judge the baseline's fairness."""
    return {
        "statement": ("UNTUNED reference baseline. Architecture and training protocol are fixed textbook "
                      "defaults chosen before any results were seen and applied unchanged at every curve "
                      "point. No architecture or hyperparameter search, no augmentation. Trained on the "
                      "identical seeded splits and evaluated on the identical fixed test set as the "
                      "classical models. It is NOT a state-of-the-art CNN; a tuned or larger network could "
                      "score higher, and the comparison should be read as 'plain CNN vs plain-but-careful "
                      "classical pipeline'."),
        "config": asdict(C),
        "framework": f"PyTorch {torch.__version__} (CPU)",
        "input": f"preprocessed {config.PREPROCESS.image_size}x{config.PREPROCESS.image_size} binary images, "
                 "scaled to [0, 1]",
    }
