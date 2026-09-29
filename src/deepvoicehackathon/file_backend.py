"""External-data-only FILE fusion with training-fold normalization and fixed export.

This does not alter the submission runner. Group CV is development selection;
its selected minimum is not an unbiased estimate of generalization.
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import official_eer

SCORE_FIELDS = (
    "df_voice_mean", "df_music_mean", "df_music_max", "spectra", "nii",
    "artifactnet", "lofcz", "current_voice", "current_music", "voice_present",
    "music_present", "current_file", "file_music",
)
INTERACTIONS = (("voice_present", "df_voice_mean"), ("voice_present", "current_voice"),
                ("music_present", "file_music"), ("music_present", "current_music"))
FEATURES = tuple("logit:" + name for name in SCORE_FIELDS) + tuple(
    "logit:" + left + "*" + right for left, right in INTERACTIONS)
C_VALUES = (0.03, 0.3, 3.0)
EPSILON = 1e-6
SEED = 20260905
FIXED_SCORES = ("current_file", "file_w025", "file_w05", "file_w1", "file_or_w0", "file_or_w1")


def features(rows):
    if not rows:
        raise ValueError("empty feature input")
    values = np.asarray([[float(row[name]) for name in SCORE_FIELDS] for row in rows])
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("component probabilities must be finite and in [0, 1]")
    indices = {name: index for index, name in enumerate(SCORE_FIELDS)}
    products = np.column_stack([values[:, indices[a]] * values[:, indices[b]] for a, b in INTERACTIONS])
    bounded = np.clip(np.column_stack((values, products)), EPSILON, 1 - EPSILON)
    return np.log(bounded) - np.log1p(-bounded)


def identities(rows):
    """Validate cache provenance and identify actual source overlap, not just IDs."""
    if not rows:
        raise ValueError("empty score cache")
    result = {name: set() for name in ("sample_id", "group", "speaker", "content_id", "parent_sha256", "audio_sha256", "pipeline")}
    owners = defaultdict(set)
    for row in rows:
        if row["sample_id"] in result["sample_id"]:
            raise ValueError("duplicate sample_id")
        if row["status"] != "ok" or json.loads(row["fallbacks_json"]) or json.loads(row["errors_json"]):
            raise ValueError("training/selection cache has an inference failure")
        if str(row["file_fake"]) not in ("0", "1"):
            raise ValueError("trusted binary FILE labels required")
        provenance = json.loads(row["provenance_json"])
        for name in ("sample_id", "group", "audio_sha256"):
            if not row[name]:
                raise ValueError(f"missing {name}")
            result[name].add(row[name])
        for name in ("speaker", "content_id"):
            value = provenance.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing source {name}")
            result[name].add(value)
            owners[(name, value)].add(row["group"])
        parents = json.loads(provenance["parents_json"])
        if not parents or any(not parent.get("sha256") for parent in parents):
            raise ValueError("parent audio hashes required")
        result["parent_sha256"].update(parent["sha256"] for parent in parents)
        for parent in parents:
            owners[("parent_sha256", parent["sha256"])].add(row["group"])
        owners[("audio_sha256", row["audio_sha256"])].add(row["group"])
        if not row["pipeline_fingerprint"]:
            raise ValueError("missing pipeline identity")
        result["pipeline"].add(row["pipeline_fingerprint"])
    if any(len(groups) != 1 for groups in owners.values()):
        raise ValueError("speaker/content/audio appears in multiple CV groups")
    if len(result["pipeline"]) != 1:
        raise ValueError("mixed inference pipelines")
    features(rows)
    return result


def assert_disjoint(development, validation):
    left, right = identities(development), identities(validation)
    if left["pipeline"] != right["pipeline"]:
        raise ValueError("development/validation pipeline mismatch")
    overlaps = {name: sorted(left[name] & right[name]) for name in left if name != "pipeline"}
    overlaps["all_audio_content"] = sorted(
        (left["parent_sha256"] | left["audio_sha256"]) & (right["parent_sha256"] | right["audio_sha256"]))
    if any(overlaps.values()):
        raise ValueError("development/validation source overlap: " + json.dumps(overlaps))
    if {int(row["file_fake"]) for row in validation} != {0, 1}:
        raise ValueError("both selection classes required")
    return {name: {"development": len(left[name]), "validation": len(right[name]), "overlap": 0}
            for name in left if name != "pipeline"} | {"all_audio_content": {"overlap": 0}}


def binary_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.round(np.asarray(scores, dtype=float), 10)
    if len(labels) != len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("invalid metric arrays")
    both = set(labels) == {0, 1}
    return {"n": len(labels), "n_real": int(sum(labels == 0)), "n_fake": int(sum(labels == 1)),
            "eer": official_eer(labels, scores) if both else None,
            "auc": float(roc_auc_score(labels, scores)) if both else None}


def fit_backend(rows):
    identity = identities(rows)
    if len(identity["group"]) < 4:
        raise ValueError("at least four independent development groups required")
    x = features(rows)
    y = np.asarray([int(row["file_fake"]) for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    if set(y) != {0, 1}:
        raise ValueError("both development classes required")
    folds = list(GroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(x, y, groups))
    fold_ids = np.full(len(rows), -1, dtype=int)
    for fold, (train, valid) in enumerate(folds):
        if set(y[train]) != {0, 1} or set(y[valid]) != {0, 1}:
            raise ValueError("every training/OOF fold needs both classes")
        fold_ids[valid] = fold

    def pipeline(c):
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000, random_state=SEED, solver="lbfgs"))

    results, predictions = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for c in C_VALUES:
            oof = np.full(len(rows), np.nan)
            for train, valid in folds:
                estimator = pipeline(c).fit(x[train], y[train])
                oof[valid] = estimator.predict_proba(x[valid])[:, 1]
            results.append({"C": c, **binary_metrics(y, oof)})
            predictions[c] = oof
        best = min(results, key=lambda result: (result["eer"], -result["auc"], result["C"]))
        fitted = pipeline(best["C"]).fit(x, y)
    scaler, classifier = fitted.steps[0][1], fitted.steps[1][1]
    model = {"version": 1, "task": "FILE_FAKE only", "feature_names": list(FEATURES),
             "epsilon": EPSILON, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
             "coefficient": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
             "C": best["C"], "class_weight": "balanced", "seed": SEED,
             "pipeline_fingerprint": next(iter(identity["pipeline"])),
             "training_rows": len(rows), "training_groups": len(identity["group"])}
    parity = float(np.max(np.abs(predict_backend(model, rows) - fitted.predict_proba(x)[:, 1])))
    if parity > 1e-12:
        raise ValueError("exported scorer differs from sklearn")
    report = {"search": results, "selected_C": best["C"], "export_max_abs_error": parity,
              "fold_by_sample_id": {row["sample_id"]: int(fold) for row, fold in zip(rows, fold_ids)},
              "development_oof": compare(rows, predictions[best["C"]]),
              "limits": "Selected OOF metric is development evidence, not nested-CV or hidden-score estimate."}
    return model, report


def predict_backend(model, rows):
    if model["version"] != 1 or model["feature_names"] != list(FEATURES) or model["epsilon"] != EPSILON:
        raise ValueError("incompatible fixed backend contract")
    mean, scale, coef = (np.asarray(model[name], dtype=float) for name in ("mean", "scale", "coefficient"))
    if any(value.shape != (len(FEATURES),) or not np.isfinite(value).all() for value in (mean, scale, coef)):
        raise ValueError("invalid fixed coefficients")
    if (scale <= 0).any() or not np.isfinite(model["intercept"]):
        raise ValueError("invalid scale/intercept")
    # Only immutable training statistics; no input-batch moments or adaptation.
    return expit(((features(rows) - mean) / scale) @ coef + model["intercept"])


def compare(rows, learned):
    if len(rows) != len(learned):
        raise ValueError("prediction length mismatch")
    y = np.asarray([int(row["file_fake"]) for row in rows])
    scores = {name: np.asarray([float(row[name]) for row in rows]) for name in FIXED_SCORES}
    scores["learned"] = np.asarray(learned)

    def summarize(indices):
        return {"groups": len({rows[i]["group"] for i in indices}),
                "metrics": {name: binary_metrics(y[indices], values[indices]) for name, values in scores.items()}}

    domains = sorted({row["domain"] for row in rows})
    generators = [json.loads(row["provenance_json"])["music_generator"] for row in rows]
    return {"overall": summarize(np.arange(len(rows))),
            "by_domain": {name: summarize(np.asarray([i for i, row in enumerate(rows) if row["domain"] == name])) for name in domains},
            "by_music_generator_block": {name: summarize(np.asarray([i for i, value in enumerate(generators) if value == name]))
                                         for name in sorted(set(generators))},
            "generator_note": "Block-associated generator slices; not generator-held-out CV or causal attribution."}
