"""Train and test the main model: SVM + SHAP-guided look-alike specialists.

Objectives 2, 3 and 4. Everything is chosen on TRAINING data using
out-of-fold predictions (which pairs, which regions, which specialists to keep).
The test set is only scored at the very end.

Usage:  python scripts/run_lookalike.py
Output: models/main_svm_lookalike.joblib, results/metrics/lookalike.json,
        results/metrics/per_class_final.json, results/metrics/confused_pairs_final.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src import config, data_loader, evaluate, lookalike  # noqa: E402
from src.explain import build_groups  # noqa: E402
from src.features import fusion  # noqa: E402
from src.utils import get_logger, run_cli, save_json, set_seed  # noqa: E402

log = get_logger("run_lookalike")
C = config.LOOKALIKE
MET = config.PATHS.metrics


def oof_scores(X, y, folds):
    """Out-of-fold SVM decision scores (aligned to sorted class names)."""
    classes = np.unique(y)
    out = np.zeros((len(y), len(classes)))
    for k, (tr, te) in enumerate(folds):
        pre, clf = lookalike.fit_svm(X[tr], y[tr])
        out[te] = clf.decision_function(pre.transform(X[te]))
        log.info("fold %d: out-of-fold accuracy %.2f%%", k + 1, 100 * (classes[out[te].argmax(1)] == y[te]).mean())
    return out, classes


def test_pair_oof(pair, X, y, first, second, folds, groups):
    """On training images whose top two are this pair: specialist vs SVM, out of fold."""
    a, b = pair
    reached = (((first == a) & (second == b)) | ((first == b) & (second == a))) & np.isin(y, pair)
    svm_right = spec_right = total = 0
    for tr, te in folds:
        use = np.intersect1d(te, np.flatnonzero(reached))
        if not len(use):
            continue
        sp = lookalike.fit_specialist(pair, X[tr], y[tr], groups)
        svm_right += int((first[use] == y[use]).sum())
        spec_right += int((sp.predict(X[use]) == y[use]).sum())
        total += len(use)
    return {"pair": [str(a), str(b)], "images": total, "svm_right": svm_right, "specialist_right": spec_right}


def mcnemar(y, base, new):
    """Exact two-sided McNemar test on paired predictions."""
    broken = int(((base == y) & (new != y)).sum())
    fixed = int(((new == y) & (base != y)).sum())
    return {"fixed": fixed, "broken": broken, "p_value": float(binomtest(fixed, fixed + broken, 0.5).pvalue) if fixed + broken else 1.0}


def main() -> int:
    set_seed()
    index = data_loader.load_processed_index()
    index["_pos"] = np.arange(len(index))
    train, test, _ = data_loader.stratified_split(index, config.SAMPLING.main_train_per_class, quiet=True)
    X = fusion.load_feature_set(index, "fused")
    Xtr, ytr = X[train["_pos"].to_numpy()], train["label"].to_numpy().astype(str)
    Xte, yte = X[test["_pos"].to_numpy()], test["label"].to_numpy().astype(str)
    groups = build_groups("fused")

    # 1. out-of-fold guesses on the training set
    folds = list(StratifiedKFold(C.folds, shuffle=True, random_state=config.SEED).split(Xtr, ytr))
    scores, classes = oof_scores(Xtr, ytr, folds)
    order = np.argsort(scores, axis=1)[:, ::-1]
    first, second = classes[order[:, 0]], classes[order[:, 1]]

    # 2. look-alike pairs: the most confused pairs in the training set
    counts: dict[tuple[str, str], int] = {}
    for t, p in zip(ytr[first != ytr], first[first != ytr]):
        k = tuple(sorted((str(t), str(p))))
        counts[k] = counts.get(k, 0) + 1
    candidates = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:C.top_pairs]]
    log.info("candidate pairs: %s", candidates)

    # 3. keep a specialist only if it beats the SVM out of fold
    checks = Parallel(n_jobs=C.n_jobs)(delayed(test_pair_oof)(p, Xtr, ytr, first, second, folds, groups) for p in candidates)
    kept = [tuple(c["pair"]) for c in checks if c["specialist_right"] > c["svm_right"]]
    log.info("kept %d of %d specialists", len(kept), len(candidates))

    # 4. final model on the whole training set
    pre, svm = lookalike.fit_svm(Xtr, ytr)
    specs = Parallel(n_jobs=C.n_jobs)(delayed(lookalike.fit_specialist)(p, Xtr, ytr, groups) for p in kept)
    model = lookalike.LookalikeModel(pre, svm, lookalike.fit_calibrator(scores, ytr), np.asarray(svm.classes_),
                                     {s.pair: s for s in specs}, info={"n_train": int(len(ytr))})
    lookalike.save(model)

    # 5. the test set, once
    d = model.decide(Xte)
    vote = svm.predict(pre.transform(Xte))             # the SVM's ordinary one-vs-one vote
    acc = lambda p: float((p == yte).mean())
    fired = np.array([u is not None for u in d["specialist"]])
    per_pair = []
    for a, b in kept:
        m = np.isin(yte, (a, b))
        hit = np.array([u == (a, b) for u in d["specialist"]])
        sp = model.specialists[(a, b)]
        per_pair.append({"pair": [a, b], "regions": sp.regions, "images_consulted": int(hit.sum()),
                         "errors_svm": int((d["svm"][m] != yte[m]).sum()), "errors_final": int((d["final"][m] != yte[m]).sum()),
                         "correct_svm": int((d["svm"][hit] == yte[hit]).sum()), "correct_final": int((d["final"][hit] == yte[hit]).sum())})
    cats = dict(zip(index["label"], index["category"]))
    cat_of = np.array([cats[l] for l in yte])
    by_cat = {c: {"svm": acc_cat, "final": float((d["final"][cat_of == c] == yte[cat_of == c]).mean())}
              for c in np.unique(cat_of) for acc_cat in [float((d["svm"][cat_of == c] == yte[cat_of == c]).mean())]}
    result = {
        "method": "SVM on LDA features; for known look-alike pairs a binary SVM on SHAP-chosen regions decides",
        "train_per_class": config.SAMPLING.main_train_per_class, "n_test": int(len(yte)),
        "settings": {"top_pairs": C.top_pairs, "top_regions": C.top_regions, "shap_images": C.shap_images,
                     "shap_samples": C.shap_samples, "folds": C.folds, "specialist_C": C.specialist_C},
        "candidate_pairs": [list(p) for p in candidates], "training_checks": checks, "pairs_kept": [list(p) for p in kept],
        "accuracy": {"svm_vote": acc(vote), "svm_score": acc(d["svm"]), "final": acc(d["final"])},
        "macro_f1_final": float(evaluate.summarize(yte, d["final"], cat_of)["macro_f1"]),
        "by_category": by_cat,
        "images_consulted": int(fired.sum()),
        "on_consulted": {"svm_correct": int((d["svm"][fired] == yte[fired]).sum()), "final_correct": int((d["final"][fired] == yte[fired]).sum())},
        "against_svm_score": mcnemar(yte, d["svm"], d["final"]), "against_svm_vote": mcnemar(yte, vote, d["final"]),
        "per_pair": per_pair,
    }
    save_json(result, MET / "lookalike.json")

    labels = np.asarray(svm.classes_)
    cm = evaluate.confusion(yte, d["final"], labels)
    table = evaluate.per_class_table(yte, d["final"], labels, cats)
    save_json({"model": "SVM with look-alike specialists", "classes": table.to_dict(orient="records")}, MET / "per_class_final.json")
    pairs = evaluate.most_confused_pairs(cm, labels)
    save_json({"model": "SVM with look-alike specialists", "total_errors": int((d["final"] != yte).sum()), "n_test": int(len(yte)),
               "pairs": pairs.to_dict(orient="records")}, MET / "confused_pairs_final.json")
    log.info("TEST: SVM vote %.2f%%, SVM score %.2f%%, with specialists %.2f%% | %s | %s", 100 * acc(vote), 100 * acc(d["svm"]),
             100 * acc(d["final"]), result["against_svm_score"], result["against_svm_vote"])
    return 0


if __name__ == "__main__":
    run_cli(main)
