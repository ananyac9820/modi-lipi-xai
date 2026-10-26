"""Untuned reference CNN for the learning-curve comparison.

Objective 4 (synopsis): the simple CNN we compare against, trained on exactly the same splits.

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
from pathlib import Path
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


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomic write, so a power cut mid-save never leaves a corrupt checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _fit(n_classes: int, X: torch.Tensor, y: torch.Tensor, epochs: int, seed: int,
         X_val: torch.Tensor | None = None, y_val: torch.Tensor | None = None,
         ckpt: Path | None = None) -> tuple[nn.Module, list[dict[str, float]], int]:
    """Train for up to ``epochs``; with validation data, early-stop on val loss.

    With ``ckpt``, the complete training state (weights, optimiser, both RNG
    states, early-stopping bookkeeping, history) is saved after every epoch,
    and an existing checkpoint is resumed. Training on CPU is deterministic,
    so a resumed run gives bit-for-bit the same model as an uninterrupted one;
    an interruption costs at most the epoch in progress.

    Returns (model with best weights, history, best_epoch).
    """
    seed_everything(seed)
    model = PlainCNN(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=C.learning_rate)
    gen = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    best, best_epoch, best_state, waited, start, finished = np.inf, epochs, None, 0, 1, False
    if ckpt is not None and ckpt.exists():
        s = torch.load(ckpt, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        gen.set_state(s["gen"])
        torch.set_rng_state(s["torch_rng"])
        history, best, best_epoch, best_state, waited = s["history"], s["best"], s["best_epoch"], s["best_state"], s["waited"]
        start, finished = len(history) + 1, s["finished"]
        log.info("    resumed from checkpoint after epoch %d%s", len(history), " (phase already finished)" if finished else "")
    for ep in range(start, epochs + 1):
        if finished:
            break
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
        rec["wall_s"] = round(time.perf_counter() - t, 2)          # incl. validation; used for resumed timing
        history.append(rec)
        log.info("    epoch %2d  train_loss %.4f%s  (%.1fs)", ep, tr_loss,
                 f"  val_loss {rec['val_loss']:.4f}" if "val_loss" in rec else "", rec["seconds"])
        finished = ep == epochs or (X_val is not None and waited >= C.early_stopping_patience)
        if ckpt is not None:
            _save_checkpoint(ckpt, {"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                                    "torch_rng": torch.get_rng_state(), "history": history, "best": best,
                                    "best_epoch": best_epoch, "best_state": best_state, "waited": waited,
                                    "finished": finished})
        if X_val is not None and waited >= C.early_stopping_patience:
            log.info("    early stop: no val improvement for %d epochs (best epoch %d)", waited, best_epoch)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def _sleep_corrected(wall: float, *histories: list[dict[str, float]]) -> tuple[float, int]:
    """Training time with laptop sleep removed.

    The timer keeps running while the computer sleeps, so an epoch that spans
    a sleep looks enormously long. Any epoch longer than 10x the median epoch
    of its phase is counted at the median instead. Accuracy is unaffected:
    training is deterministic and sleep only pauses it.
    """
    excess, n = 0.0, 0
    for hist in histories:
        if not hist:
            continue
        med = float(np.median([e["seconds"] for e in hist]))
        for e in hist:
            if e["seconds"] > 10 * med:
                excess += e["seconds"] - med
                n += 1
    return wall - excess, n


def _epoch_sum_time(*histories: list[dict[str, float]]) -> tuple[float, int]:
    """Training time for a run split over several sessions: the sum of the
    per-epoch wall times (training + validation), sleep-corrected the same way
    as ``_sleep_corrected``. Wall clock is meaningless across sessions."""
    total, n = 0.0, 0
    for hist in histories:
        if not hist:
            continue
        times = [e.get("wall_s", e["seconds"]) for e in hist]
        med = float(np.median(times))
        for s in times:
            if s > 10 * med:
                s, n = med, n + 1
            total += s
    return total, n


def train_and_evaluate(train_imgs: np.ndarray, train_labels: np.ndarray, classes: list[str],
                       seed: int = config.SEED, ckpt_dir: Path | None = None) -> tuple[nn.Module, dict[str, Any]]:
    """Fixed protocol: early stopping on a train-only validation slice, then
    (optionally) refit on the full training set for the selected epoch count.

    ``ckpt_dir``: if given, each phase checkpoints every epoch there and
    resumes from it (see ``_fit``); the caller deletes it once the point is saved.

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

    ck1 = ckpt_dir / "phase1.pt" if ckpt_dir else None
    ck2 = ckpt_dir / "phase2.pt" if ckpt_dir else None
    resumed = bool(ck1 and ck1.exists())
    t0 = time.perf_counter()
    log.info("  phase 1: early stopping (train %d / val %d)", int((~val).sum()), int(val.sum()))
    model, hist1, best_epoch = _fit(len(classes), X_all[~val], torch.from_numpy(y_all[~val]), C.max_epochs, seed,
                                    X_all[val], torch.from_numpy(y_all[val]), ckpt=ck1)
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
        model, hist2, _ = _fit(len(classes), X_all, torch.from_numpy(y_all), best_epoch, seed, ckpt=ck2)
    wall = time.perf_counter() - t0
    if resumed:
        train_time, slept = _epoch_sum_time(hist1, hist2)
    else:
        train_time, slept = _sleep_corrected(wall, hist1, hist2)
    info = {
        "train_time_s": round(train_time, 2),
        "wall_clock_s": round(wall, 2),
        "resumed_from_checkpoint": resumed,
        "train_time_method": "sum of per-epoch times (run split over sessions)" if resumed
        else "wall clock minus sleep",
        "epochs_spanning_sleep": slept,
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
