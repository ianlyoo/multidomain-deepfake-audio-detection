"""Strict FILE33 with cross-head consistency features; callers own whole-raw55 fallback."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from . import component_support_file_backend as baseline
VERSION = 15
MODEL_TYPE = "gradient_boosted_trees_file33_consistency"
SCORE_FIELDS = baseline.SCORE_FIELDS
EPSILON = baseline.EPSILON
FEATURES = (*baseline.FEATURES, "logit:file29", "logit:abs(voice38-music17)",
            "logit:voice38*(1-voice_present)", "logit:music17*(1-music_present)")
PROTOCOL_SHA = "b3cc93cc038df1f8132df2b4abc109359b9006d6b754194b89e73c6b42ae7ff6"
PROTOCOL_COMMIT = "37ac1bd"
SOURCE55_SHA = "5ef3c3ce76cf06270e76e6498c395939d1ebcf828e46e4ce75f2a672a650f78d"
PREDECESSOR_SHA = "0f38b7c9006d8ac309c6157140f7c9b473b390939cecbb9b1bb91282d1e08c92"
STAGE1_SHA = "0f38b7c9006d8ac309c6157140f7c9b473b390939cecbb9b1bb91282d1e08c92"
RECIPE = "fixed64-depth2-lr0.05-leaf24-file33-consistency"
IMPLEMENTATIONS = ("src/deepvoicehackathon/consistency_file_backend.py",
                   "tools/train_sub65_consistency_file.py")
RECEIPT_HASHES = ("rows_sha256", "groups_sha256", "labels_sha256", "weights_sha256",
                  "features_sha256", "suppliers_sha256")
def _is_real_probability(value):
    if isinstance(value, (bool, np.bool_)):
        return False
    if not np.isscalar(value):
        return False
    if np.asarray(value).dtype.kind not in "fiu":
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(number) and 0.0 <= number <= 1.0)
def _logit(value):
    clipped = min(max(float(value), EPSILON), 1.0 - EPSILON)
    return float(np.log(clipped) - np.log1p(-clipped))
def features(rows):
    try:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("nonempty consistency FILE rows required")
        original = baseline.features(rows)
        if original.shape != (len(rows), 29):
            raise ValueError("component-support29 feature width drift")
        added = []
        for row in rows:
            for key in ("voice_present", "music_present", "music17", "voice38", "file29"):
                if key not in row:
                    raise ValueError("missing consistency probability: " + key)
                if not _is_real_probability(row[key]):
                    raise ValueError("finite real consistency probability required: " + key)
            v = round(float(row["voice38"]), 10)
            m = round(float(row["music17"]), 10)
            f = round(float(row["file29"]), 10)
            vp = round(float(row["voice_present"]), 10)
            mp = round(float(row["music_present"]), 10)
            nova = round(v * (1.0 - vp), 10)
            noma = round(m * (1.0 - mp), 10)
            diff = round(abs(v - m), 10)
            for candidate in (f, diff, nova, noma):
                if not np.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
                    raise ValueError("rounded consistency support out of range")
            added.append([_logit(f), _logit(diff), _logit(nova), _logit(noma)])
        return np.column_stack((original, np.asarray(added, dtype=np.float64)))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid FILE33 consistency evidence") from error
def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task="FILE_FAKE only",
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype="float32",
        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
        loss="log_loss", subsample=1., seed=20260905, training_rows=2316, training_groups=40,
        extra_training_rows=396, native_training_rows=264, fake_training_rows=132,
        protocol_sha256=PROTOCOL_SHA, protocol_commit=PROTOCOL_COMMIT,
        source55_sha256=SOURCE55_SHA, predecessor_sha256=PREDECESSOR_SHA,
        stage1_model_sha256=STAGE1_SHA, stage1_oof_seed=20260905, inner_group_seed=20260915,
        recipe=RECIPE)
def _validate_receipt(receipt):
    return baseline._validate_receipt(receipt)
def validate_model(model):
    try:
        fixed = model_metadata()
        if type(model) is not dict or set(model) != set(fixed) | {"initial_log_odds", "trees", "implementation_sha256", "fit_receipt"}:
            raise ValueError("strict consistency FILE33 model required")
        for key, value in fixed.items():
            if type(model[key]) is not type(value) or model[key] != value:
                raise ValueError("fixed FILE33 recipe mismatch: " + key)
        pins = model["implementation_sha256"]
        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any(not baseline._sha(v) for v in pins.values()):
            raise ValueError("portable implementation SHA256 map required")
        _validate_receipt(model["fit_receipt"])
        if type(model["initial_log_odds"]) not in (int, float) or not np.isfinite(model["initial_log_odds"]):
            raise ValueError("finite initial log odds required")
        if type(model["trees"]) is not list or len(model["trees"]) != 64:
            raise ValueError("exactly64 trees required")
        names = ("children_left", "children_right", "feature", "threshold", "value")
        for tree in model["trees"]:
            if type(tree) is not dict or set(tree) != set(names):
                raise ValueError("strict tree object required")
            arrays = [tree[k] for k in names]
            if any(type(a) is not list for a in arrays):
                raise ValueError("JSON tree arrays required")
            left, right, feature, threshold, value = arrays
            n = len(left)
            if not 1 <= n <= 7 or any(len(a) != n for a in arrays):
                raise ValueError("depth-two tree bounds required")
            if any(type(v) is not int for a in arrays[:3] for v in a):
                raise ValueError("integer tree coordinates required")
            if any(type(v) not in (int, float) or not np.isfinite(v) for a in arrays[3:] for v in a):
                raise ValueError("finite numeric tree data required")
            seen, pending = set(), [(0, 0)]
            while pending:
                node, depth = pending.pop()
                if node in seen or depth > 2:
                    raise ValueError("shared/cyclic/deep node")
                seen.add(node)
                if left[node] == -1:
                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:
                        raise ValueError("invalid sklearn leaf")
                else:
                    if not 0 <= feature[node] < 33 or not node < left[node] < n or not node < right[node] < n:
                        raise ValueError("invalid split coordinates")
                    pending.extend(((left[node], depth+1), (right[node], depth+1)))
            if len(seen) != n:
                raise ValueError("unreachable tree node")
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError("malformed FILE33 consistency model") from error
def predict(model, rows):
    validate_model(model)
    x = features(rows).astype(np.float32)
    raw = np.full(len(rows), model["initial_log_odds"], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for tree in model["trees"]:
            for i, values in enumerate(x):
                node = 0
                while tree["children_left"][node] != -1:
                    node = (tree["children_left"][node] if float(values[tree["feature"][node]]) <= tree["threshold"][node]
                            else tree["children_right"][node])
                raw[i] += model["learning_rate"] * tree["value"][node]
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite FILE33 log odds")
    return expit(raw)
