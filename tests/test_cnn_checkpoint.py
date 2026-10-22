"""The CNN baseline can be stopped and resumed without changing its result."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.baselines import cnn_baseline as cb  # noqa: E402


class _Interrupt(Exception):
    pass


def _data(n: int = 48, classes: int = 3):
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.integers(0, 256, (n, 1, 64, 64), dtype=np.uint8))
    y = torch.from_numpy(np.arange(n) % classes)
    return X, y


def test_resume_is_bit_identical(tmp_path, monkeypatch):
    X, y = _data()
    Xv, yv = X[:12], y[:12]
    ref, ref_hist, ref_best = cb._fit(3, X, y, 4, seed=1, X_val=Xv, y_val=yv)

    real = cb._run_epoch
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:                      # "power cut" during epoch 3
            raise _Interrupt
        return real(*a, **k)

    ckpt = tmp_path / "phase1.pt"
    monkeypatch.setattr(cb, "_run_epoch", flaky)
    with pytest.raises(_Interrupt):
        cb._fit(3, X, y, 4, seed=1, X_val=Xv, y_val=yv, ckpt=ckpt)
    monkeypatch.setattr(cb, "_run_epoch", real)
    assert ckpt.exists()

    resumed, hist, best = cb._fit(3, X, y, 4, seed=1, X_val=Xv, y_val=yv, ckpt=ckpt)
    assert best == ref_best
    assert [h["train_loss"] for h in hist] == [h["train_loss"] for h in ref_hist]
    for k, v in ref.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), k
